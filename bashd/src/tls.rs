//! The mutual-TLS server config for [`server::serve`](crate::server::serve).
//!
//! `bashd` presents the per-container server cert the host minted and **requires**
//! a client cert chaining to the same per-container CA — so only the host that
//! generated the PKI (and holds the client key) can drive the daemon, even
//! though the published port is reachable by any process on the host. The PKI
//! arrives over stdin as a [`TlsServerMaterial`]; see the host's `pki` module.
//!
//! The crypto provider is wired explicitly (aws-lc-rs) rather than via rustls's
//! process-default, so it stays unambiguous even when a test build also links
//! reqwest's rustls backend. Flipping to `ring` is a one-line change here.

use std::sync::Arc;

use misanthropic::tool::bash::TlsServerMaterial;
use rustls::ServerConfig;
use rustls::server::WebPkiClientVerifier;

/// Build the mutual-TLS [`ServerConfig`] from the host-supplied [`TlsServerMaterial`]:
/// present the server chain, and require a client cert verified against the CA.
pub fn server_config(
    tls: &TlsServerMaterial,
) -> std::io::Result<Arc<ServerConfig>> {
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());

    let certs = rustls_pemfile::certs(&mut tls.cert_chain_pem.as_bytes())
        .collect::<Result<Vec<_>, _>>()?;
    let key = rustls_pemfile::private_key(&mut tls.key_pem.as_bytes())?
        .ok_or_else(|| invalid("no private key in bashd server material"))?;

    let mut roots = rustls::RootCertStore::empty();
    for ca in rustls_pemfile::certs(&mut tls.ca_pem.as_bytes()) {
        roots
            .add(ca?)
            .map_err(|e| invalid(&format!("bad CA cert: {e}")))?;
    }
    let verifier = WebPkiClientVerifier::builder_with_provider(
        Arc::new(roots),
        provider.clone(),
    )
    .build()
    .map_err(|e| invalid(&format!("client verifier: {e}")))?;

    let mut config = ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| invalid(&format!("tls versions: {e}")))?
        .with_client_cert_verifier(verifier)
        .with_single_cert(certs, key)
        .map_err(|e| invalid(&format!("server cert/key: {e}")))?;
    // axum-server's `from_config` does not negotiate ALPN; set it or HTTP/1.1
    // clients can stall the handshake.
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(Arc::new(config))
}

/// An `InvalidData` I/O error with `msg`.
fn invalid(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

#[cfg(test)]
mod tests {
    use axum::routing::get;
    use misanthropic::tool::bash::TlsServerMaterial;
    use rcgen::{
        CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
        SanType,
    };

    use super::server_config;

    /// Mint a CA + a server leaf (SAN 127.0.0.1) + a client leaf, returning the
    /// server material bashd consumes and the host's client identity + CA PEMs.
    fn pki() -> (TlsServerMaterial, String, String) {
        let ca_key = KeyPair::generate().unwrap();
        let ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        let ca_cert = ca_params.self_signed(&ca_key).unwrap();
        let ca_pem = ca_cert.pem();
        let issuer = Issuer::new(ca_params, ca_key);

        let leaf = |eku, san: Option<SanType>| {
            let key = KeyPair::generate().unwrap();
            let mut p = CertificateParams::new(Vec::<String>::new()).unwrap();
            p.is_ca = IsCa::NoCa;
            p.extended_key_usages = vec![eku];
            p.subject_alt_names = san.into_iter().collect();
            let cert = p.signed_by(&key, &issuer).unwrap();
            (cert.pem(), key.serialize_pem())
        };
        let (srv_cert, srv_key) = leaf(
            ExtendedKeyUsagePurpose::ServerAuth,
            Some(SanType::IpAddress(std::net::Ipv4Addr::LOCALHOST.into())),
        );
        let (cli_cert, cli_key) =
            leaf(ExtendedKeyUsagePurpose::ClientAuth, None);

        let material = TlsServerMaterial {
            cert_chain_pem: format!("{srv_cert}{ca_pem}"),
            key_pem: srv_key,
            ca_pem: ca_pem.clone(),
        };
        (material, format!("{cli_key}{cli_cert}"), ca_pem)
    }

    /// End-to-end: serve a trivial route over the real axum-server mTLS path; a
    /// client presenting the matching identity succeeds, one without is refused.
    #[tokio::test]
    async fn mutual_tls_requires_the_client_cert() {
        // Unambiguous process default for any bare `builder()` (e.g. reqwest's).
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let (material, client_identity, ca_pem) = pki();

        let config = server_config(&material).expect("server config");
        let tls = axum_server::tls_rustls::RustlsConfig::from_config(config);

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let app = axum::Router::new().route("/", get(|| async { "ok" }));
        tokio::spawn(async move {
            axum_server::from_tcp_rustls(listener, tls)
                .unwrap()
                .serve(app.into_make_service())
                .await
                .unwrap();
        });

        let url = format!("https://127.0.0.1:{}/", addr.port());

        // With the matching client identity → handshake + 200.
        let pinned = reqwest::Client::builder()
            .add_root_certificate(
                reqwest::Certificate::from_pem(ca_pem.as_bytes()).unwrap(),
            )
            .tls_built_in_root_certs(false)
            .identity(
                reqwest::Identity::from_pem(client_identity.as_bytes())
                    .unwrap(),
            )
            .build()
            .unwrap();
        // Small retry: the spawned server may not be bound on the first poll.
        let mut body = None;
        for _ in 0..50 {
            if let Ok(resp) = pinned.get(&url).send().await {
                body = Some(resp.text().await.unwrap());
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert_eq!(body.as_deref(), Some("ok"), "pinned client should connect");

        // Trusting the CA but presenting *no* client cert → rejected.
        let no_cert = reqwest::Client::builder()
            .add_root_certificate(
                reqwest::Certificate::from_pem(ca_pem.as_bytes()).unwrap(),
            )
            .tls_built_in_root_certs(false)
            .build()
            .unwrap();
        assert!(
            no_cert.get(&url).send().await.is_err(),
            "a client without the cert must be refused"
        );
    }
}

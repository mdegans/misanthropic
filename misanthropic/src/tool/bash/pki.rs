//! Ephemeral per-container PKI for the [`DockerSandbox`](super::DockerSandbox) ↔
//! `bashd` mutual-TLS channel.
//!
//! [`Pki::generate`] mints a throwaway CA and two leaves on every sandbox start:
//! a **server** cert (SAN `127.0.0.1`, handed to `bashd` over stdin as a
//! [`TlsServerMaterial`]) and a **client** cert (kept host-side, loaded into the
//! pinned `reqwest` client). Nothing touches disk — every PEM lives in
//! [`Zeroizing`] memory — and the CA private key is dropped the instant the
//! leaves are signed, so neither half can mint further certs even if read.
//!
//! Security split: the container only ever sees the server identity + the CA
//! *public* cert, so the sandboxed payload (same user as `bashd`) can at most
//! impersonate `bashd`, never forge the client cert that authorizes the control
//! plane. See [`super::docker`] for the stdin injection.

use rcgen::{
    BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer,
    KeyPair, KeyUsagePurpose, SanType,
};
use time::{Duration, OffsetDateTime};
use zeroize::Zeroizing;

use super::{BashError, TlsServerMaterial};

/// How long the ephemeral leaves stay valid. Generous on purpose: the security
/// comes from per-container, never-on-disk keys, not from a tight expiry — and a
/// session must outlive a multi-day agent run without the cert lapsing under it.
/// Sessions older than this should be recycled regardless.
const VALIDITY: Duration = Duration::days(30);

/// A generated per-container PKI: the server half handed to `bashd`, plus the
/// host's own client identity and trust root for the pinned `reqwest` client.
pub struct Pki {
    /// Server cert chain + key + CA, injected into `bashd` over stdin.
    pub server: TlsServerMaterial,
    /// PEM: the host's client key followed by its client cert — the `reqwest`
    /// [`Identity`](reqwest::Identity). Never leaves the host.
    pub client_identity_pem: Zeroizing<String>,
    /// PEM: the CA cert the host trusts `bashd`'s server cert against.
    pub ca_pem: Zeroizing<String>,
}

impl Pki {
    /// Generate a fresh CA and the server/client leaves. The CA private key is
    /// dropped before returning, so neither half can mint further certs.
    pub fn generate() -> Result<Self, BashError> {
        let now = OffsetDateTime::now_utc();
        // A small backdate absorbs host/container clock skew (≈0 under Docker,
        // which shares the host clock, but free insurance).
        let not_before = now - Duration::hours(1);
        let not_after = now + VALIDITY;

        // Ephemeral CA.
        let ca_key = KeyPair::generate().map_err(tls_err)?;
        let mut ca =
            CertificateParams::new(Vec::<String>::new()).map_err(tls_err)?;
        ca.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
        ca.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::DigitalSignature,
        ];
        ca.not_before = not_before;
        ca.not_after = not_after;
        let ca_cert = ca.self_signed(&ca_key).map_err(tls_err)?;
        let ca_pem = Zeroizing::new(ca_cert.pem());
        // Consumes the CA params + key; the key is unreachable from here on and
        // drops with `issuer` at the end of this function.
        let issuer = Issuer::new(ca, ca_key);

        // Server leaf — SAN 127.0.0.1 (what the host dials in every network
        // mode), serverAuth.
        let loopback = SanType::IpAddress(std::net::Ipv4Addr::LOCALHOST.into());
        let server = leaf(
            &issuer,
            (not_before, not_after),
            ExtendedKeyUsagePurpose::ServerAuth,
            Some(loopback),
        )?;
        let server = TlsServerMaterial {
            cert_chain_pem: format!("{}{}", *server.cert_pem, *ca_pem),
            key_pem: server.key_pem.to_string(),
            ca_pem: ca_pem.to_string(),
        };

        // Client leaf — clientAuth, no SAN (auth is by chain, not name).
        let client = leaf(
            &issuer,
            (not_before, not_after),
            ExtendedKeyUsagePurpose::ClientAuth,
            None,
        )?;
        let client_identity_pem =
            Zeroizing::new(format!("{}{}", *client.key_pem, *client.cert_pem));

        Ok(Self {
            server,
            client_identity_pem,
            ca_pem,
        })
    }
}

/// A signed leaf: its public cert PEM and (zeroized) private-key PEM.
struct Leaf {
    cert_pem: Zeroizing<String>,
    key_pem: Zeroizing<String>,
}

/// Sign a non-CA leaf under `issuer` with the given EKU and optional SAN.
fn leaf(
    issuer: &Issuer<'_, KeyPair>,
    (not_before, not_after): (OffsetDateTime, OffsetDateTime),
    eku: ExtendedKeyUsagePurpose,
    san: Option<SanType>,
) -> Result<Leaf, BashError> {
    let key = KeyPair::generate().map_err(tls_err)?;
    let mut p =
        CertificateParams::new(Vec::<String>::new()).map_err(tls_err)?;
    p.is_ca = IsCa::NoCa;
    p.extended_key_usages = vec![eku];
    p.subject_alt_names = san.into_iter().collect();
    p.not_before = not_before;
    p.not_after = not_after;
    let cert = p.signed_by(&key, issuer).map_err(tls_err)?;
    Ok(Leaf {
        cert_pem: Zeroizing::new(cert.pem()),
        key_pem: Zeroizing::new(key.serialize_pem()),
    })
}

/// Map an [`rcgen::Error`] into a [`BashError`].
fn tls_err(e: rcgen::Error) -> BashError {
    BashError::Backend(format!("sandbox TLS cert generation failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generated material is well-formed: `reqwest` accepts the client identity
    /// and the CA root, and `bashd`'s server half carries a key + a chain. (The
    /// full mutual-TLS handshake is exercised in `bashd`'s own `tls` tests.)
    #[test]
    fn generated_pki_loads_into_reqwest() {
        let pki = Pki::generate().expect("generate pki");

        // The host client half parses as a usable reqwest identity + root.
        reqwest::Identity::from_pem(pki.client_identity_pem.as_bytes())
            .expect("client identity loads");
        reqwest::Certificate::from_pem(pki.ca_pem.as_bytes())
            .expect("ca cert loads");
        reqwest::Client::builder()
            .add_root_certificate(
                reqwest::Certificate::from_pem(pki.ca_pem.as_bytes()).unwrap(),
            )
            .tls_built_in_root_certs(false)
            .identity(
                reqwest::Identity::from_pem(pki.client_identity_pem.as_bytes())
                    .unwrap(),
            )
            .build()
            .expect("pinned mTLS client builds");

        // The server half bound for bashd carries a key and a two-cert chain
        // (leaf + CA).
        assert!(pki.server.key_pem.contains("PRIVATE KEY"));
        assert_eq!(
            pki.server
                .cert_chain_pem
                .matches("BEGIN CERTIFICATE")
                .count(),
            2,
            "server chain should be leaf + CA"
        );
        assert!(pki.server.ca_pem.contains("BEGIN CERTIFICATE"));
    }
}

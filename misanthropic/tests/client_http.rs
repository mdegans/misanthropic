//! Offline tests of [`Client`]'s HTTP path: a local [`wiremock`] server stands
//! in for the API (via [`Client::base_url`]) and replays captured wire
//! fixtures, so request building, headers, error mapping, and SSE parsing
//! run on every commit without a key. The live suite still guards the wire
//! itself; this guards everything between it and the caller.
#![cfg(all(feature = "client", feature = "batch"))]

use futures::TryStreamExt;
use misanthropic::{
    CachedPrompt, Client, Prompt, Transport,
    batch::{Batch, Id, Status},
    client::{AnthropicError, Error},
    prompt::message::Role,
    stream::FilterExt,
};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{body_partial_json, header, method, path, query_param},
};

/// Keys are validated only for length.
const KEY_LEN: usize = 108;

fn key() -> String {
    "k".repeat(KEY_LEN)
}

const MESSAGE: &str =
    include_str!("../test/data/system_after_server_tool.response.json");
const THINKING_SSE: &str = include_str!("../test/data/thinking.sse.stream.txt");

const BATCH_ID: &str = "msgbatch_013Zva2CMHLNnXjNJJKqJ2EF";

/// A server, and a [`Client`] pointed at it.
async fn setup() -> (MockServer, Client) {
    let server = MockServer::start().await;
    let client = Client::new(key()).unwrap().base_url(server.uri()).unwrap();
    (server, client)
}

fn prompt() -> Prompt {
    Prompt::default().add_message((Role::User, "Hi")).unwrap()
}

/// The JSON body of the `n`th request the server received.
async fn sent(server: &MockServer, n: usize) -> Value {
    let requests: Vec<Request> = server.received_requests().await.unwrap();
    serde_json::from_slice(&requests[n].body).unwrap()
}

/// An Anthropic error body, as the API wraps it.
fn error_body(kind: &str, message: &str) -> Value {
    json!({ "type": "error", "error": { "type": kind, "message": message } })
}

fn batch_meta(status: &str, results_url: Option<String>) -> Value {
    json!({
        "id": BATCH_ID,
        "type": "message_batch",
        "processing_status": status,
        "request_counts": {
            "processing": 0,
            "succeeded": 1,
            "errored": 1,
            "canceled": 0,
            "expired": 0
        },
        "ended_at": null,
        "created_at": "2024-08-20T18:37:24.100435Z",
        "expires_at": "2024-08-21T18:37:24.100435Z",
        "archived_at": null,
        "cancel_initiated_at": null,
        "results_url": results_url,
    })
}

#[tokio::test]
async fn message_replays_captured_response() {
    let (server, client) = setup().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", key().as_str()))
        .and(header("anthropic-version", Client::ANTHROPIC_VERSION))
        .and(header("content-type", "application/json"))
        .and(body_partial_json(json!({ "stream": false })))
        .respond_with(ResponseTemplate::new(200).set_body_string(MESSAGE))
        .expect(1)
        .mount(&server)
        .await;

    let message = client.message(prompt()).await.unwrap();

    assert_eq!(message.id, "msg_01MrmMsHowupXYxTKoBoqs9e");
    assert!(message.to_string().contains("httpbin.org/anything/1"));
    assert_eq!(
        sent(&server, 0).await["messages"][0]["content"][0]["text"],
        "Hi"
    );
}

#[tokio::test]
async fn beta_header_rides_every_request() {
    let (server, client) = setup().await;
    let client = client.beta(Client::INTERLEAVED_THINKING_BETA);
    Mock::given(header("anthropic-beta", Client::INTERLEAVED_THINKING_BETA))
        .respond_with(ResponseTemplate::new(200).set_body_string(MESSAGE))
        .expect(1)
        .mount(&server)
        .await;

    client.message(prompt()).await.unwrap();
}

#[tokio::test]
async fn stream_replays_captured_sse() {
    let (server, client) = setup().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({ "stream": true })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(THINKING_SSE),
        )
        .expect(1)
        .mount(&server)
        .await;

    let stream = client.stream(prompt()).await.unwrap();
    let text: String = stream.text().try_collect().await.unwrap();

    assert_eq!(text, "27 * 453 = 12,231");
}

#[tokio::test]
async fn request_follows_the_prompts_stream_flag() {
    let (server, client) = setup().await;
    Mock::given(body_partial_json(json!({ "stream": true })))
        .respond_with(ResponseTemplate::new(200).set_body_string(THINKING_SSE))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(MESSAGE))
        .mount(&server)
        .await;

    let streaming = prompt().stream();
    assert!(matches!(
        client.request(&streaming).await.unwrap(),
        misanthropic::Response::Stream { .. }
    ));
    assert!(matches!(
        client.request(prompt()).await.unwrap(),
        misanthropic::Response::Message { .. }
    ));
}

#[tokio::test]
async fn rate_limit_carries_retry_after() {
    let (server, client) = setup().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "7")
                .set_body_json(error_body("rate_limit_error", "slow down")),
        )
        .mount(&server)
        .await;

    let err = client.message(prompt()).await.unwrap_err();

    let Error::Anthropic(err) = err else {
        panic!("expected an Anthropic error, got {err:?}");
    };
    assert_eq!(err.retry_after(), Some(std::time::Duration::from_secs(7)));
    assert!(matches!(err, AnthropicError::RateLimit { .. }));
}

#[tokio::test]
async fn overloaded_without_retry_after() {
    let (server, client) = setup().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(529)
                .set_body_json(error_body("overloaded_error", "busy")),
        )
        .mount(&server)
        .await;

    let Err(err) = client.stream(prompt()).await else {
        panic!("expected an error");
    };

    assert!(matches!(
        err,
        Error::Anthropic(AnthropicError::Overloaded {
            retry_after: None,
            ..
        })
    ));
}

#[tokio::test]
async fn unknown_error_type_keeps_http_status() {
    let (server, client) = setup().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(418)
                .set_body_json(error_body("teapot_error", "short and stout")),
        )
        .mount(&server)
        .await;

    let Error::Anthropic(err) = client.message(prompt()).await.unwrap_err()
    else {
        panic!("expected an Anthropic error");
    };

    assert_eq!(err.status().map(u16::from), Some(418));
    assert!(err.to_string().contains("teapot_error: short and stout"));
}

#[tokio::test]
async fn non_json_error_body_is_surfaced() {
    let (server, client) = setup().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(503)
                .set_body_string("<html>Service Unavailable</html>"),
        )
        .mount(&server)
        .await;

    let err = client.message(prompt()).await.unwrap_err();

    match err {
        Error::NonJsonResponse { status, body } => {
            assert_eq!(status, 503);
            assert_eq!(body, "<html>Service Unavailable</html>");
        }
        other => panic!("expected NonJsonResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn unparseable_ok_body_is_a_parse_error() {
    let (server, client) = setup().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{\"nope\""))
        .mount(&server)
        .await;

    assert!(matches!(
        client.message(prompt()).await.unwrap_err(),
        Error::Parse(_)
    ));
}

#[tokio::test]
async fn models_lists_and_maps_errors() {
    let (server, client) = setup().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(query_param("limit", "1000"))
        .and(header("x-api-key", key().as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{
                "type": "model",
                "id": "claude-haiku-4-5",
                "display_name": "Claude Haiku 4.5",
                "created_at": "2025-10-01T00:00:00Z"
            }],
            "has_more": false
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    let models = client.models().await.unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].display_name, "Claude Haiku 4.5");

    // Through `Transport`, too — the chat driver's view of a `Client`.
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("[]"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    let err = Transport::<Prompt>::models(&client).await.unwrap_err();
    assert!(matches!(err, Error::Parse(_)));

    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_json(error_body("authentication_error", "bad key")),
        )
        .mount(&server)
        .await;
    assert!(matches!(
        Transport::<CachedPrompt>::models(&client)
            .await
            .unwrap_err(),
        Error::Anthropic(AnthropicError::Authentication { .. })
    ));
}

#[tokio::test]
async fn transport_send_is_a_message() {
    let (server, client) = setup().await;
    Mock::given(body_partial_json(json!({ "stream": false })))
        .respond_with(ResponseTemplate::new(200).set_body_string(MESSAGE))
        .expect(2)
        .mount(&server)
        .await;

    let prompt = prompt();
    let plain = Transport::send(&client, &prompt).await.unwrap();
    let cached = CachedPrompt::from(prompt);
    let cached = Transport::send(&client, &cached).await.unwrap();

    assert_eq!(plain.id, cached.id);
}

#[tokio::test]
async fn count_tokens_strips_generation_fields() {
    let (server, client) = setup().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/count_tokens"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "input_tokens": 42 })),
        )
        .expect(1)
        .mount(&server)
        .await;

    let prompt = prompt().temperature(0.5).stream();
    assert_eq!(client.count_tokens(&prompt).await.unwrap(), 42);

    let body = sent(&server, 0).await;
    let body = body.as_object().unwrap();
    assert!(body.contains_key("messages"));
    for field in ["max_tokens", "temperature", "stream"] {
        assert!(!body.contains_key(field), "{field} reached count_tokens");
    }
}

#[tokio::test]
async fn count_tokens_reports_a_bad_body() {
    let (server, client) = setup().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "tokens": 42 })),
        )
        .mount(&server)
        .await;

    assert!(matches!(
        client.count_tokens(prompt()).await.unwrap_err(),
        Error::Parse(_)
    ));
}

#[tokio::test]
async fn batch_submit_poll_and_collect() {
    let (server, client) = setup().await;
    let results_url =
        format!("{}/v1/messages/batches/{BATCH_ID}/results", server.uri());

    // Submit.
    Mock::given(method("POST"))
        .and(path("/v1/messages/batches/"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(batch_meta("in_progress", None)),
        )
        .expect(2)
        .mount(&server)
        .await;

    let pending = client
        .tagged_batch([(Id::default(), prompt()), (Id::default(), prompt())])
        .await
        .unwrap();
    assert!(matches!(pending.status(), Status::InProgress));
    let submitted = sent(&server, 0).await;
    assert_eq!(submitted["requests"].as_array().unwrap().len(), 2);

    // Untagged submission takes the same path.
    client.batch([prompt()]).await.unwrap();

    // Poll: still in progress.
    Mock::given(method("GET"))
        .and(path(format!("/v1/messages/batches/{BATCH_ID}")))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(batch_meta("in_progress", None)),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    let Ok(Batch::Pending(pending)) = client.batch_poll(pending).await else {
        panic!("expected the batch to still be pending");
    };

    // Poll: ended, with results — one success, one error, one line for an
    // id we never sent (dropped), and one garbage line (skipped).
    Mock::given(method("GET"))
        .and(path(format!("/v1/messages/batches/{BATCH_ID}")))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(batch_meta("ended", Some(results_url))),
        )
        .mount(&server)
        .await;
    let ids: Vec<_> =
        pending.prompts().into_iter().map(|(id, _)| *id).collect();
    let jsonl = [
        json!({
            "custom_id": ids[0].to_string(),
            "result": {
                "type": "succeeded",
                "message": serde_json::from_str::<Value>(MESSAGE).unwrap(),
            }
        }),
        json!({
            "custom_id": ids[1].to_string(),
            "result": {
                "type": "errored",
                "error": error_body("invalid_request_error", "too long"),
            }
        }),
        json!({
            "custom_id": "00000000-0000-0000-0000-000000000000",
            "result": { "type": "canceled" }
        }),
    ]
    .iter()
    .map(Value::to_string)
    .chain(["not json".to_string()])
    .collect::<Vec<_>>()
    .join("\n");
    Mock::given(method("GET"))
        .and(path(format!("/v1/messages/batches/{BATCH_ID}/results")))
        .respond_with(ResponseTemplate::new(200).set_body_string(jsonl))
        .mount(&server)
        .await;

    let Ok(Batch::Ready(mut ready)) = client.batch_poll(pending).await else {
        panic!("expected the batch to be ready");
    };
    assert_eq!(ready.remove_ok().len(), 1);
    assert_eq!(ready.remove_errors().len(), 1);
    assert_eq!(ready.iter().count(), 0);
}

#[tokio::test]
async fn batch_poll_failure_hands_the_batch_back() {
    let (server, client) = setup().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(batch_meta("in_progress", None)),
        )
        .mount(&server)
        .await;
    let pending = client.batch([prompt()]).await.unwrap();

    // A gateway error first, then a body that isn't a batch.
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(502).set_body_string("Bad Gateway"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    let Err(err) = client.batch_poll(pending).await else {
        panic!("expected the poll to fail");
    };
    assert!(matches!(
        err.client_error,
        Error::NonJsonResponse { status: 502, .. }
    ));

    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&server)
        .await;
    let Err(err) = client.batch_poll(err.into_pending()).await else {
        panic!("expected the poll to fail");
    };
    let (err, pending) = err.decompose();
    assert!(matches!(err, Error::Parse(_)));
    assert_eq!(pending.meta().id, BATCH_ID);
}

#[tokio::test]
async fn batch_submit_error_is_mapped() {
    let (server, client) = setup().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_json(error_body("invalid_request_error", "empty")),
        )
        .mount(&server)
        .await;

    assert!(matches!(
        client.batch(Vec::<Prompt>::new()).await.unwrap_err(),
        Error::Anthropic(AnthropicError::InvalidRequest { .. })
    ));
}

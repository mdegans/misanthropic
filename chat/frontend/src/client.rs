use eventsource_stream::Eventsource;
use misanthropic::{
    exports::futures::{Stream, StreamExt},
    prompt::TurnOrderError,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsValue;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum Event {
    TurnOrderError(TurnOrderError),
    Prompt(misanthropic::prompt::Prompt<'static>),
    MisanthropicClientError(String),
    Stream(misanthropic::stream::Event),
    StreamError(String),
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),
    #[error(transparent)]
    SerdeWasmBindgen(#[from] serde_wasm_bindgen::Error),
    #[error(transparent)]
    EventsourceStream(
        #[from] eventsource_stream::EventStreamError<reqwest::Error>,
    ),
}

#[derive(Clone)]
pub struct Client {
    client: reqwest::Client,
}

impl Client {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    pub async fn stream(
        &self,
    ) -> Result<impl Stream<Item = Result<Event, Error>>, reqwest::Error> {
        let events = self
            .client
            .get("http://localhost:8080/events")
            .send()
            .await?
            .bytes_stream()
            .eventsource();
        Ok(events.map(|event: Result<eventsource_stream::Event, _>| {
            let event = event?;
            // get a JsValue from the event's data
            let val = JsValue::from_str(&event.data);
            let event: Event = serde_wasm_bindgen::from_value(val)?;

            Ok(event)
        }))
    }
}

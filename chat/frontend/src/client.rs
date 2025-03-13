use eventsource_stream::Eventsource;
use misanthropic::exports::futures::{Stream, StreamExt};
use model::{request::Request, response::Response};
use serde::Serialize;

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
    #[error("JsValue error: {0:?}")]
    JsValue(wasm_bindgen::JsValue),
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
    ) -> Result<
        impl Stream<Item = Result<Response, Error>>,
        Box<dyn std::error::Error>,
    > {
        let get = self.client.get("http://localhost:8079/events");

        let events = get.send().await?.bytes_stream().eventsource();

        Ok(events.map(|event: Result<eventsource_stream::Event, _>| {
            log::trace!("EVENT: {:?}", event);

            let event = event?;
            // get a JsValue from the event's data
            let val = js_sys::JSON::parse(&event.data)
                .map_err(|e| Error::JsValue(e.into()))?;
            let event: Response = serde_wasm_bindgen::from_value(val)?;

            Ok(event)
        }))
    }

    pub async fn send<'a, T>(&self, request: T) -> Result<(), reqwest::Error>
    where
        T: Into<Request>,
    {
        let request: Request = request.into();
        self.client
            .post("http://localhost:8079/message")
            .json(&request)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}

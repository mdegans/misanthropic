use axum::{
    routing::{get, post},
    Router,
};
use endpoints::create_router;
use shuttle_runtime::SecretStore;
use std::sync::Arc;
use tokio::sync::Mutex;

mod endpoints;
mod prompt;
mod state;

pub(crate) use state::AppState;

type UserMessage = misanthropic::prompt::message::UserMessage<'static>;
type Prompt = misanthropic::Prompt<'static>;
type Message = misanthropic::prompt::message::Message<'static>;

#[shuttle_runtime::main]
async fn main(
    #[shuttle_runtime::Secrets] secrets: SecretStore,
) -> shuttle_axum::ShuttleAxum {
    let router = create_router(secrets);

    Ok(router.into())
}

#[cfg(test)]
mod tests {}

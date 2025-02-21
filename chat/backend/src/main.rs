use endpoints::create_router;
use shuttle_runtime::SecretStore;

mod endpoints;
mod prompt;
mod state;

pub(crate) use state::AppState;

type UserMessage = misanthropic::prompt::message::UserMessage<'static>;
type Prompt = misanthropic::Prompt<'static>;

#[shuttle_runtime::main]
async fn main(
    #[shuttle_runtime::Secrets] secrets: SecretStore,
) -> shuttle_axum::ShuttleAxum {
    let router = create_router(secrets);

    Ok(router.into())
}

#[cfg(test)]
mod tests {}

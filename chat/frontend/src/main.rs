use dioxus::prelude::*;

use components::Navbar;
use views::Chat;

mod client;
mod components;
mod utils;
mod views;

#[derive(Debug, Clone, Routable, PartialEq)]
#[rustfmt::skip]
enum Route {
    #[layout(Navbar)]
    #[route("/chat/")]
    Chat {},
}

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/styling/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");

fn main() {
    wasm_logger::init(wasm_logger::Config::default());

    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    rsx! {
        // Global app resources
        document::Link { rel: "icon", href: FAVICON }
        document::Stylesheet { href: MAIN_CSS }
        document::Stylesheet { href: TAILWIND_CSS }

        document::Meta {
            name: "viewport",
            content: "width=device-width, initial-scale=1.0",
        }

        Router::<Route> {}
    }
}

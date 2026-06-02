/// The browser's `localStorage`, if available (it isn't in every context,
/// e.g. some private-mode configurations).
fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

/// Read a string from `localStorage` by `key`, or [`None`] if absent or
/// unavailable.
pub fn storage_get(key: &str) -> Option<String> {
    local_storage()?.get_item(key).ok().flatten()
}

/// Write a string to `localStorage` under `key`. A no-op (with a warning) if
/// storage is unavailable or the write is rejected (e.g. quota exceeded).
///
/// Unlike `dioxus_sdk`'s reactive `use_storage`, this writes synchronously and
/// eagerly — no spawned watcher task to (fail to) flush. See issue #66.
pub fn storage_set(key: &str, value: &str) {
    match local_storage() {
        Some(storage) => {
            if let Err(e) = storage.set_item(key, value) {
                log::warn!("Failed to write `{key}` to localStorage: {e:?}");
            }
        }
        None => log::warn!("localStorage unavailable; `{key}` not persisted."),
    }
}

/// https://users.rust-lang.org/t/async-sleep-in-rust-wasm32/78218/6
pub async fn sleep_ms(delay: i32) {
    let mut cb = |resolve: js_sys::Function, _reject: js_sys::Function| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                &resolve, delay,
            )
            .unwrap();
    };

    let p = js_sys::Promise::new(&mut cb);

    wasm_bindgen_futures::JsFuture::from(p).await.unwrap();
}

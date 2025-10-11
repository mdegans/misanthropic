use crate::prompt::message::Content;

/// A trait for rendering objects for the [`Navigator`] agent.
pub trait RenderForNavigator {
    /// Render the object for the [`Navigator`] agent.
    fn render_for_navigator(&self) -> Content;
}

/// A trait for converting objects to JSON suitable for the [`Navigator`] agent.
pub trait NavigatorJson {
    /// Convert the object to JSON suitable for the [`Navigator`] agent. This
    /// usually involves stripping out any fields that the navigator does not
    /// need to see.
    fn navigator_json(&self) -> serde_json::Value;
}

/// A trait for rendering objects for the [`Architect`] agent.
pub trait RenderForArchitect {
    /// Render the object for the [`Architect`] agent.
    fn render_for_architect(&self) -> Content;
}

/// A trait for rendering objects for the [`Janitor`] agent.
pub trait RenderForJanitor {
    /// Render the object for the [`Janitor`] agent.
    fn render_for_janitor(&self) -> Content;
}

/// A trait for rendering objects for the primary agent using the memory palace.
pub trait RenderForPrimaryAgent {
    /// Render the object for the primary agent using the memory palace.
    fn render_for_primary_agent(&self) -> Content;
}

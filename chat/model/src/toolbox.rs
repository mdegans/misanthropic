use misanthropic::tool::ToolBox;

pub fn create() -> ToolBox {
    ToolBox::new().add_typed(misanthropic::tool::Notepad::new())
}

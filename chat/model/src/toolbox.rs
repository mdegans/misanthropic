use misanthropic::tool::ToolBox;

pub fn create() -> ToolBox {
    ToolBox::new().add(misanthropic::tool::Notepad::new())
}

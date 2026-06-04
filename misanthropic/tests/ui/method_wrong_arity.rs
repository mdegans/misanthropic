//! A `#[method]` fn must take exactly one argument besides `self`.
#![allow(unused)]
use misanthropic::prompt::message::Content;
use misanthropic::tool::tool;

struct Bad;

#[tool]
impl Bad {
    /// Two args is not allowed.
    #[method]
    async fn oops(
        &mut self,
        a: String,
        b: String,
    ) -> Result<Content, Content> {
        let _ = (a, b);
        Ok("".into())
    }
}

fn main() {}

pub mod engine;
pub mod operator;
pub mod probe;
pub mod signal;

trait Wrapper<T> {
    fn get(&self) -> &T;
}

use std::sync::{Condvar, Mutex};

pub struct Event(Mutex<bool>, Condvar);

impl Event {
    pub fn new() -> Self {
        Event(Mutex::new(false), Condvar::new())
    }

    pub fn clear(&self) {
        self.set_value(false);
    }

    pub fn set(&self) {
        self.set_value(true);
    }

    fn set_value(&self, value: bool) {
        let Event(lock, cvar) = self;
        let mut finished = lock.lock().unwrap();
        if *finished != value {
            *finished = value;
            cvar.notify_all();
        }
    }

    pub fn wait(&self) {
        let Event(lock, cvar) = self;
        let mut finished = lock.lock().unwrap();

        while !*finished {
            finished = cvar.wait(finished).unwrap();
        }
    }
}

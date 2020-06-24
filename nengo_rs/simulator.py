class Simulator:
    def __init__(self, network, dt=0.001, seed=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        pass

    def run(self, time_in_seconds):
        pass

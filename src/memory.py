class ZoraMemory:
    def __init__(self):
        self.history = []

    def store_interaction(self, user_input, model_response):
        self.history.append((user_input, model_response))

    def get_last_response(self):
        return self.history[-1] if self.history else None

memory = ZoraMemory()
class Broken:
    """Does not implement the protocol: conformance must reject it."""

    name = "broken"

    def ingest(self, artifacts, checkpoint=0):
        pass

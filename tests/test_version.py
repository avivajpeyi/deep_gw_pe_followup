from deep_gw_pe_followup.version import __version__


class TestVersion:
    def test_version(self):
        assert isinstance(__version__, str)

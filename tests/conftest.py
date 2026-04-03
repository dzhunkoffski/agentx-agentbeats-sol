def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        action="store",
        default=None,
        help="Base URL of the running agent for integration tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as requiring a live agent (--agent-url)"
    )

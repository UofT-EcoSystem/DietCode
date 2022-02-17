"""
pytest Extra Options
"""

def pytest_addoption(parser):
    parser.addoption('--sched_log_fname', action="store", type=str)
    parser.addoption('-B', action="store", type=int, default=16)
    parser.addoption('--NH', action="store", type=int, default=12)
    parser.addoption('-T', action="store", type=int, default=128)
    parser.addoption('-H', action="store", type=int, default=768)
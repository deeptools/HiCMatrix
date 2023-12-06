import logging

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

logging.getLogger('cooler').setLevel(logging.WARNING)

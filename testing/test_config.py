from v7p3r_config import v7p3rConfig

# Test with centipawn_test_config
config = v7p3rConfig('configs/centipawn_test_config.json')
engine_config = config.get_engine_config()
print(f'search_algorithm from centipawn_test_config: {engine_config.get("search_algorithm", "not found")}')

# Test with default config
default_config = v7p3rConfig()
default_engine_config = default_config.get_engine_config()
print(f'search_algorithm from default_config: {default_engine_config.get("search_algorithm", "not found")}')

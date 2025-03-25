

if 'SDX_TEST_DIR' in os.environ:
    base_path = os.getenv('SDX_TEST_DIR')
else:
    base_path = os.getcwd()
syn_path = os.path.join(base_path, 'synDatasets')
attack_path = os.path.join(base_path, 'suppress_attacks')
os.makedirs(attack_path, exist_ok=True)
import sys, shutil, csv, json
from pathlib import Path
sys.path.insert(0, '.')

from src.utils.storage import StorageManager
from src.submission.package import run_submission_packaging

tmp = Path('tmp_test_pkg')
tmp.mkdir(exist_ok=True)
try:
    config = {
        'dataset': {'files': {
            'test_source1': 'test_source1.tsv',
            'test_source2': 'test_source2.tsv',
            'test_source3': 'test_source3.tsv',
        }, 'separator': '\t', 'encoding': 'utf-8'},
    }
    storage = StorageManager('configs/config.yaml')
    storage.data_root = tmp / 'data'
    storage.storage_root = tmp / 'artifacts_root'
    storage.raw_dir = storage.data_root / 'raw'
    storage.splits_dir = storage.data_root / 'splits'
    storage.features_dir = storage.data_root / 'features'
    storage.checkpoints_dir = storage.storage_root / 'checkpoints'
    storage.experiments_dir = storage.storage_root / 'experiments'
    storage.artifacts_dir = storage.storage_root / 'artifacts'
    storage.output_dir = storage.storage_root / 'output'
    storage.initialize_directories()

    with open(storage.raw_dir / 'test_source1.tsv', 'w') as f:
        f.write('entity_id\tbusiness_name\n')
        f.write('s1_001\tAlpha Corp\n')
        f.write('s1_002\tBeta Inc\n')
        f.write('s1_003\tGamma LLC\n')

    with open(storage.output_dir / 'matching_results.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['s1_entity_id', 'matching_ids'])
        writer.writerow(['s1_001', 's2_001'])
        writer.writerow(['s1_002', ''])
        writer.writerow(['s1_003', 's3_001'])

    with open(storage.output_dir / 'candidate_pairs.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['s1_entity_id', 'candidate_ids'])
        writer.writerow(['s1_001', 's2_001,s2_002'])
        writer.writerow(['s1_002', 's2_003'])
        writer.writerow(['s1_003', 's3_001,s3_002'])

    report = run_submission_packaging(config, storage)
    issues = report['issues']
    assert report['valid'] == True, str(issues)
    assert report['stats']['num_entities'] == 3
    assert report['stats']['num_singletons'] == 1
    print('Stage 15 Test Passed!')
finally:
    shutil.rmtree(tmp, ignore_errors=True)

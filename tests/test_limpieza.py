import json
import numpy as np
import pytest
from src import limpieza


def test_exact_ratios_and_reproducibility():
    groups = [[i] for i in range(100)]
    result = limpieza.split_groups(groups)
    assert [len(result[s]) for s in ('train', 'val', 'test')] == [70, 15, 15]
    assert result == limpieza.split_groups(groups)


def test_transitive_groups_remain_together():
    groups = limpieza._grupos(100, np.array([0, 1]), np.array([1, 2]))
    result = limpieza.split_groups(groups)
    owners = {i: name for name, items in result.items() for i in items}
    assert owners[0] == owners[1] == owners[2]
    assert sorted(i for items in result.values() for i in items) == list(range(100))


def test_cleaning_preserves_bridge_after_deduplication(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    hashes = rng.integers(0, 2, (102, 256)).astype(bool)
    # A~B duplicate, B~C similar, A !~ C. Removing B must not split A/C.
    hashes[0] = False
    hashes[1] = False
    hashes[1, :8] = True
    hashes[2] = False
    hashes[2, :30] = True
    monkeypatch.setattr(limpieza, '_procesar_imagen', lambda task: ('ok', hashes[int(task[0])]))
    data = {s: {'rutas': [], 'etiquetas': []} for s in ('train', 'val', 'test')}
    data['train'] = {'rutas': [str(i) for i in range(102)], 'etiquetas': [None]*102}
    result, report = limpieza.limpiar_dataset(data, ['humo', 'fuego'], tmp_path)
    manifest = json.loads((tmp_path/'manifiesto.json').read_text())
    owners = {item['origen']: name for name, items in manifest['splits'].items() for item in items}
    assert '1' not in owners
    assert owners['0'] == owners['2']
    assert sum(len(items['rutas']) for items in result.values()) == 101
    assert manifest['cross_split_similar_pairs'] == 0
    group_owners = {}
    for name, items in manifest['splits'].items():
        for item in items:
            assert group_owners.setdefault(item['group_id'], name) == name
    assert sum(manifest['actual_fractions'].values()) == pytest.approx(1)


def test_invalid_splits_rejected():
    with pytest.raises(ValueError):
        limpieza.split_groups([[0], [1]])
    with pytest.raises(ValueError):
        limpieza.split_groups([[0], [1], [2]], .7, .4)

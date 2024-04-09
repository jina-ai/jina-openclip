MTEB_TASKS_TO_METRICS = {
    'classification': 'accuracy',
    'clustering': 'v-measure',
    'pair-classification': 'average-precision',
    'reranking': 'mean-average-precision',
    'retrieval': 'ndcg@10',
    'sts': 'spearman-correlation',
    'summarization': 'spearman-correlation',
}

CLIPB_METRICS = {
    'acc1': 'accuracy@1',
    'acc5': 'accuracy@5',
    'mean_per_class_recall': 'mean-per-class-recall',
    'mean_average_precision': 'mean-average-precision',
    'image_retrieval_recall@5': 'image-retrieval-recall@5',
    'text_retrieval_recall@5': 'text-retrieval-recall@5',
}


class Row:

    def __init__(self):
        self.benchmark = 'benchmark'
        self.task = 'task'
        self.language = 'en'
        self.dataset = 'dataset'
        self.model = 'model'
        self.metrics = {}


def _read_clipb_rows(fname: str, model: str):
    with open(fname, 'r') as f:
        cliplines = f.read().splitlines()

    clipdata = [list(line.split(',')) for line in cliplines]

    header = clipdata[0]
    metrics = {}
    task_idx = header.index('task')
    dataset_idx = header.index('dataset')
    language_idx = header.index('language') if 'language' in header else None
    for i, field in enumerate(header):
        if field in CLIPB_METRICS:
            metrics[CLIPB_METRICS[field]] = i

    rows = []
    for row in clipdata[1:]:
        _row_metrics = {}
        for metric, idx in metrics.items():
            v = row[idx]
            if v and v != 'nan':
                _row_metrics[metric] = float(v)

        task = row[task_idx]
        dataset = row[dataset_idx].replace('wds/', '')
        language = row[language_idx] if language_idx is not None else None

        clsrow = Row()
        clsrow.benchmark = 'CLIPB'
        clsrow.task = '-'.join(task.split('_'))
        clsrow.dataset = dataset
        clsrow.model = model
        if language:
            clsrow.language = language
        clsrow.metrics = _row_metrics

        rows.append(clsrow)

    return rows


def _read_mteb_rows(fname: str, model: str):

    with open(fname, 'r') as f:
        mteblines = f.read().splitlines()

    mtebdata = [list(line.split(',')) for line in mteblines]

    rows = []
    for row in mtebdata[1:]:
        task, dataset, value = row
        value = float(value)
        dataset, lang = dataset.split(' ')
        if lang == 'average':
            continue
        assert lang.startswith('(') and lang.endswith(')')
        lang = lang[1:-1]

        _metric_name = MTEB_TASKS_TO_METRICS[task]
        clsrow = Row()
        clsrow.benchmark = 'MTEB'
        clsrow.task = '-'.join(task.split('_'))
        clsrow.language = lang
        clsrow.dataset = dataset
        clsrow.model = model
        clsrow.metrics = {_metric_name: value}
        rows.append(clsrow)

    return rows


def _format_results(rows: list):

    newrows = []
    _processed_metrics = []

    for benchmark in ['CLIPB', 'MTEB']:
        tasks = {row.task for row in rows if row.benchmark == benchmark}
        tasks = sorted(list(tasks))
        for task in tasks:
            selected = [
                row for row in rows if row.task == task and row.benchmark == benchmark
            ]
            metrics = {
                metric for row in selected for metric in row.metrics.keys()
            }
            metrics = sorted(list(metrics))
            models = {row.model for row in selected}
            models = sorted(list(models))

            selected = sorted(selected, key=lambda x: x.dataset + x.model)
            for row in selected:
                newrows.append(
                    [row.benchmark, row.task, row.dataset, row.language, row.model] +
                    [''] * len(_processed_metrics) +
                    [
                        f'{row.metrics[metric]:.5f}' if metric in row.metrics else ''
                        for metric in metrics
                    ]
                )

            for model in models:
                average = {}
                for metric in metrics:
                    values = [
                        row.metrics[metric] for row in selected
                        if row.model == model and metric in row.metrics
                    ]
                    average[metric] = sum(values) / len(values)
                newrows.append(
                    [benchmark, task, 'average', '', model] +
                    [''] * len(_processed_metrics) +
                    [f'{average[metric]:.5f}' for metric in metrics]
                )

            _processed_metrics += metrics

    header = (
        ['benchmark', 'task', 'dataset', 'language', 'model'] +
        _processed_metrics
    )
    maxlen = len(header)
    return [header] + [row + [''] * (maxlen - len(row)) for row in newrows]


def main():
    data = [
        (_read_mteb_rows, 'mteb-jclip.csv', 'jclip-v0'),
        (_read_mteb_rows, 'mteb-jembeddings.csv', 'jembeddings-v2'),
        (_read_clipb_rows, 'clipb-jclip.csv', 'jclip-v0'),
        (_read_clipb_rows, 'clipb-vitb16openai.csv', 'vit-b16-openai'),
    ]

    rows = []
    for func, fname, model in data:
        rows.extend(func(fname, model))

    rows = _format_results(rows)

    with open('results.csv', 'w') as f:
        f.write('\n'.join([','.join(row) for row in rows]))


if __name__ == '__main__':
    main()

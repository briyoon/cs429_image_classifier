from tqdm import tqdm

class progress_bar():
    """
    Wrapper for tqdm to handle setup and formatting.
    """
    def __init__(self, iterable, **kwargs):
        self.tqdm_bar = tqdm(
            iterable,
            ascii=".>=",
            bar_format="{n_fmt}/{total_fmt} [{bar:40}] - {elapsed}{postfix}",
            unit=" batches",
            **kwargs
        )

    def __iter__(self):
        return self.tqdm_bar.__iter__()

    def __next__(self):
        return self.tqdm_bar.__next__()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        self.tqdm_bar.set_postfix(ordered_dict, refresh, **kwargs)


# test
if __name__ == "__main__":
    import time
    iterable = range(200)

# bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}""

    EPOCHS = 5
    for epoch in range(1, EPOCHS + 1):
        for i in progress_bar(iterable, epoch, EPOCHS, ascii=".>=", ):
            time.sleep(0.01)
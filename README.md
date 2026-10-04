# Disk Scheduling with a Learned Selector

This is my CSE 307 Operating Systems term paper project. I implemented four disk scheduling methods: FCFS, SCAN, C-SCAN, and SSTF. I also trained a small model that looks at a batch of disk requests and picks one of these methods. The program then checks what happens when the request pattern changes from sequential to bursty.

This is a simulation. It does not change the disk scheduler in Linux or Windows.

## What you need

- Python 3.10 or newer
- The packages in `requirements.txt` (NumPy and Matplotlib)

From this folder, install the packages with:

```bash
python -m pip install -r requirements.txt
```

If `python` does not work on your Linux system, use `python3` in all the commands below.

## How to run

Run these commands from the folder that contains `experiment.py`:

```bash
python -m unittest -v test_schedulers test_experiment
python experiment.py
python plot_results.py
```

The first command runs the checks for the schedulers and the generated requests. The second runs the experiment and saves its numbers in `results/`. The last command makes the charts from those saved numbers. A full run usually takes only a short time.

If you want to save a separate experiment without changing the submitted results, use:

```bash
python experiment.py --output my_results
```

The plotting script reads from `results/`, so run `python experiment.py` without `--output` before making the submitted charts.

## What is in the project

| File | What it does |
| --- | --- |
| `disk_schedulers.py` | Implements the four scheduling methods. |
| `experiment.py` | Makes request batches, trains the selector, and saves the measurements. |
| `plot_results.py` | Makes the three charts in `results/`. |
| `test_schedulers.py`, `test_experiment.py` | Check important algorithm and workload cases. |
| `requirements.txt` | Lists the Python packages. |
| `results/summary.json` | Has the main results. |
| `results/test_batches.csv`, `results/timeline_batches.csv` | Have the results for each batch. |
| `results/*.png` | Show the comparison, shift, and confidence results. |
| `202414033_Os_Term_paper.docx` | The written report with the cover page. |

## Short summary of the results

The simulated disk has cylinders 0 to 199. Each batch has 20 requests, and the head starts at cylinder 100. I trained the selector on 900 batches, used 300 batches to adjust its confidence score, and tested it on a different 300 batches. The random seed is `3072026`, so the same code produces the same requests and results.

On the 300 test batches, the selector chose a method with the lowest seek distance 91% of the time. Its total seek distance was **46,734** cylinders. SSTF did slightly better at **45,834**. This means the model usually picked well, but some wrong picks cost enough distance to make its total worse than SSTF.

For the workload shift, the first 30 batches were sequential and the next 30 were bursty. The selector's average seek distance went from about **68.2** to **92.3** cylinders per batch. FCFS went from about **79.6** to **503.5**, so it was much more affected by the change. The selector's correct-choice rate went from **96.7%** to **93.3%**.

Seek distance is used here as a simple estimate of seek time. This project does not measure real disk latency, and it does not include rotation time, data transfer time, or fairness between requests.

## AI assistance

I chose the disk scheduling track and the NumPy approach. I used OpenAI Codex to help draft parts of the code, tests, charts, and documentation, and to help run the first experiment. I reviewed the implementation and the saved measurements for this submission. The report explains the results in plain language. I am responsible for checking and understanding the final work.

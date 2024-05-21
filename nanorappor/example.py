from response import RapporResponse
from decode import decode_rappor_responses 
from multiprocessing import Pool
import random
import numpy as np
#from scipy.optimize import lsq_linear
from scipy.stats import norm, expon

import matplotlib.pyplot as plt
import click

def run_trial(k, h, cohorts, p, q, f, marker, np_seed=42, num_reports=100_000,
              thresh=1.0):
    np.random.seed(np_seed)
    answers = np.random.exponential( 10, (num_reports,)).astype(int)
    answers = answers[answers < 50]
    lex = list(range(0, 100))
    resp = []

    for i, a in enumerate(answers):
        cohort = i % cohorts
        resp.append(RapporResponse(a, cohort, h, k, f, p, q)) 

    found, found_weights, errs = decode_rappor_responses(resp, lex,
                                                         thresh=thresh)
    print(found)
    print(np.intersect1d(found, answers))
    precision = ( len(np.intersect1d(found, answers)) / len(np.unique(found)))
    recall = len(np.intersect1d(found, answers)) / len(np.unique(answers)) 

    return [precision, recall, marker]


def run_trial_labels(k, h, cohorts, p, q, f, sz, np_seed=42, pdf='norm'):
    print(f'trials: {sz}')
    np.random.seed(np_seed)
    if pdf == 'norm':
        answers = np.random.normal( 0, 10, (sz,)).astype(int)
        lex = list(range(-30, 30))
    else:
        answers = np.random.exponential(10, (sz,)).astype(int)
        answers = answers[answers<50]
        lex = list(range(0, 100))
    resp = []

    for i, a in enumerate(answers):
        cohort = i % cohorts
        resp.append(RapporResponse(a, cohort, h, k, f, p, q)) 
    found, found_weights, errs = decode_rappor_responses(resp, lex)

    return found, found_weights, errs

@click.group()
def cli():
    pass

@cli.command()
@click.option('--output', default=None, help='output of plots')
@click.option('--num-reports', default=100_000, help='number of reports')
@click.option('--num-trials', default=10, help='num trials per hash fn count')
def num_hashes_trial(output, num_reports, num_trials):
    points = []
    markers = list(reversed(['mx', 'c+', 'y^', 'go']))

    args = []
    for fmt_i, h in enumerate([2, 4, 8, 16]):
        for _ in range(num_trials):
            args.append([128, h, 16, 0.5, 0.75, 0.5, markers[fmt_i],
                         random.randint(0, 10000000), num_reports])

    with Pool(10) as p:
        points = p.starmap(run_trial, args)

    for y, x, fmt in points:
        plt.plot(x,y,fmt)
    plt.xlabel('recall (TP / population)')
    plt.ylabel('precision (TP / (TP + FP))')
    plt.title('Varying hash functions')
    
    for fmt, h in zip(markers, [2, 4, 8, 16]):
        plt.plot([], [], fmt, label=str(h)) # workaround for plot legend
    
    plt.legend()
    if output is not None:
        plt.savefig(output, dpi=200)
    plt.show()
    
@cli.command()
@click.option('--output', default=None, help='output of plots')
@click.option('--num-reports', default=100_000, help='number of reports')
@click.option('--num-trials', default=10, help='num trials per hash fn count')
def num_cohorts_trial(output, num_reports, num_trials):
    points = []
    markers = list(reversed(['mx', 'c+', 'y^', 'go']))
    cohorts = [8, 16, 32, 64]
    args = []
    for fmt_i, m in enumerate(cohorts):
        for _ in range(num_trials):
            args.append([128, 2, m, 0.5, 0.75, 0.5, markers[fmt_i],
                         random.randint(0, 10000000), num_reports])

    with Pool(10) as p:
        points = p.starmap(run_trial, args)

    for y, x, fmt in points:
        plt.plot(x,y,fmt)
    plt.xlabel('recall (TP / population)')
    plt.ylabel('precision (TP / (TP + FP))')
    plt.title('Varying cohort size')
    
    for fmt, h in zip(markers, cohorts):
        plt.plot([], [], fmt, label=str(h)) # workaround for plot legend
    
    plt.legend()
    if output is not None:
        plt.savefig(output, dpi=200)
    plt.show()

@cli.command()
@click.option('--output', default=None, help='output of plots')
@click.option('--num-reports', default=100_000, help='number of reports')
@click.option('--num-trials', default=10, help='num trials per hash fn count')
def num_bits_trial(output, num_reports, num_trials):
    points = []
    markers = list(reversed(['mx', 'c+', 'y^', 'go']))
    ks = [128, 256, 512, 1024]
    args = []
    for fmt_i, k in enumerate(ks):
        for _ in range(num_trials):
            args.append([k, 2, 16, 0.5, 0.75, 0.5, markers[fmt_i],
                         random.randint(0, 10000000), num_reports])

    with Pool(10) as p:
        points = p.starmap(run_trial, args)

    for y, x, fmt in points:
        plt.plot(x,y,fmt)
    plt.xlabel('recall (TP / population)')
    plt.ylabel('precision (TP / (TP + FP))')
    plt.title('Varying Bloom filter size')
    
    for fmt, h in zip(markers, ks):
        plt.plot([], [], fmt, label=str(h)) # workaround for plot legend
    
    plt.legend()
    if output is not None:
        plt.savefig(output, dpi=200)
    plt.show()

@cli.command()
@click.option('--output', default=None, help='output of plots')
@click.option('--num-reports', default=100_000, help='number of reports')
@click.option('--num-trials', default=10, help='num trials per hash fn count')
def threshold_trial(output, num_reports, num_trials):
    points = []
    markers = list(reversed(['mx', 'c+', 'y^', 'go', 'k*', 'bh']))
    thresholds = [0.5, 1, 1.5, 2, 2.5, 3]
    args = []
    for fmt_i, t in enumerate(thresholds):
        for _ in range(num_trials):
            args.append([128, 2, 16, 0.5, 0.75, 0.5, markers[fmt_i],
                         random.randint(0, 10000000), num_reports, t])

    with Pool(10) as p:
        points = p.starmap(run_trial, args)

    for y, x, fmt in points:
        plt.plot(x,y,fmt)
    plt.xlabel('recall (TP / population)')
    plt.ylabel('precision (TP / (TP + FP))')
    plt.title('Varying threshold')
    
    for fmt, h in zip(markers, thresholds):
        plt.plot([], [], fmt, label=f'{h:.1f}') # workaround for plot legend
    
    plt.legend()
    if output is not None:
        plt.savefig(output, dpi=200)
    plt.show()

@cli.command()
@click.option('--output', default=None, help='output of plots')
@click.option('--pdf', default='norm', 
              type=click.Choice(['norm', 'expon'], case_sensitive=True),
              help='pdf of underyling distribution for simulated data')
def num_reports_trial(output, pdf):
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(20, 16)
    fig.tight_layout(pad=2)

    if pdf == 'norm':
        ppdf = norm.pdf
    elif pdf == 'expon':
        ppdf = expon.pdf
    else:
        raise ValueError('bad pdf')

    report_counts = [1e4, 1e5, 1e6]

    for ax, n in zip(axs, report_counts):
        found, found_weights, errs = run_trial_labels(32, 2, 16, 0.5, 0.75,
                                                      0.5, int(n), 
                                                      random.randint(0,
                                                                     1000000),
                                                      pdf=pdf)
        s = found_weights.sum()
        ax.bar(found, found_weights/s)
        ax.errorbar(found, found_weights/s, yerr=errs/s, capsize=5, ls='none')
        x = np.linspace(-50, 50, 2000)
        y = ppdf(x, scale=10) 
        ax.plot(x, y, color='gray')
        ax.fill_between(x, y, color='gray', alpha=0.5)
        ax.set_title(f'# reports = {int(n)}')
    

    if output is not None:
        plt.savefig(output, dpi=200)
    plt.show()

@cli.command()
@click.option('--output', default=None, help='output of plots')
@click.option('--pdf', default='norm', 
              type=click.Choice(['norm', 'expon'], case_sensitive=True),
              help='pdf of underyling distribution for simulated data')
def f_trials(output, pdf):
    fig, axs = plt.subplots(4, 1)
    fig.set_size_inches(20, 16)
    fig.tight_layout(pad=2)

    if pdf == 'norm':
        ppdf = norm.pdf
    elif pdf == 'expon':
        ppdf = expon.pdf
    else:
        raise ValueError('bad pdf')

    f_vals = [0, 0.25, 0.5, 0.75]
    for ax, f in zip(axs, f_vals):
        found, found_weights, errs = run_trial_labels(32, 2, 16, 0.5, 0.75,
                                                      f, 400_000, 
                                                      random.randint(0,
                                                                     1000000),
                                                      pdf=pdf)
        s = found_weights.sum()
        ax.bar(found, found_weights/s)
        ax.errorbar(found, found_weights/s, yerr=errs/s, capsize=5, ls='none')
        x = np.linspace(-50, 50, 2000)
        y = ppdf(x, scale=10) 
        ax.plot(x, y, color='gray')
        ax.fill_between(x, y, color='gray', alpha=0.5)
        ax.set_title(f'f = {f}')
    

    if output is not None:
        plt.savefig(output, dpi=200)
    plt.show()
if __name__ == '__main__':
    cli()

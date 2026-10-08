# Why this screener is shaped like this

The code is the easy part to describe. This document is the reasoning behind the
thresholds, because a screener is only as good as the discipline it encodes, and
a scoring function you can't justify is just a number generator.

---

## The premise

Most people don't lose by picking the wrong stock. They lose because they can't
sit still. **Temperament is the scarce input**, and unlike analysis, it can't be
bought or learned in a week.

So the screener is built to do the opposite of what a stock-picker wants it to
do: it is designed to **say no**. Thousands of companies go in; the job of the
quantitative stage is to remove almost all of them, and the job of the
qualitative stage is to remove more.

Corollary: every stage has to be backtestable. **A process you can't backtest is
just a story you tell yourself.**

---

## Ruler 0: the risk-free rate is the benchmark, not the index

Everything is scored against the alternative, which is the risk-free rate.

At a ~4.5% risk-free rate the equivalent P/E is about **22×**. That sets two
rules:

1. A business that isn't growing should trade at a P/E clearly **below 22**:
   its earnings yield has to beat the risk-free rate plus a risk premium.
2. A P/E **above 22** has to be justified by double-digit earnings growth. If it
   isn't, you are paying to lose to a bond.

This is why the first ruler is **E/P** (earnings yield = 1 ÷ P/E) rather than P/E
itself. P/E tells you how many flat years it takes to get your money back. E/P
tells you the return the business is printing on your capital *right now*, which
is the only thing directly comparable to a bond.

---

## The four quantitative gates

| # | Gate | Why |
|---|---|---|
| 1 | **ROE / ROIC** | Is this a machine that turns capital into profit, or a hole that consumes it? ROIC, not just ROE; debt inflates ROE without a better business. |
| 2 | **FCF conversion** | Does profit become cash? `FCF ÷ net income ≈ 100%+` means it's real. Below that, you're financing a factory or a warehouse. |
| 3 | **Gross margin** | Pricing power, and a buffer against input costs. A moat you can actually measure. |
| 4 | **E/P** | Must clear Ruler 0. |

---

## The growth and catalyst gates

A value gate on its own buys you value traps. Growth has to pay for itself:

- **PEG ≤ 1**: a high P/E can still be cheap if growth matches it.
- **Net cash**: cash minus total debt. A thick net-cash position is the floor
  under a bad year.
- **Inventory vs revenue**: inventory growing faster than revenue is a
  retail/consumer warning, not a growth signal.

---

## The qualitative gate

Numbers can't see management. But management is exactly what decides whether the
numbers repeat. So the LLM stage asks three separate questions instead of one:

| Stage | Asks | Underlying idea |
|---|---|---|
| **Grit** | Does management keep going when it's hard? | Level 5 leadership: humility + will |
| **Culture** | Does the organisation do what it says? | Culture of discipline; confronting brutal facts |
| **Catalyst** | Is there a reason *now*, and does the engine compound? | Flywheel vs doom loop; hedgehog focus |

They are split deliberately. A single blended prompt reliably produces a single
blended compliment; three separate questions force the model to take a position
on each one, and to fail one of them.

The qualitative layer never runs first. **The LLM never sees a company until the
quantitative screen has passed it.** The model is there to argue *with* the
numbers, not to replace them.

---

## The order is the design

```
① E/P: is it cheap against the risk-free rate?
    ↓
② ROE / ROIC + gross margin: real pricing power, or a value trap?
    ↓
③ FCF: cash, or a capital-expenditure hole?
    ↓
④ PEG / net cash / inventory: is growth paying for itself, and what's the floor?
    ↓
⑤ Earnings in 10 years: does the moat survive, or decay?
    ↓
⑥ Management: Level 5, discipline, flywheel. Hard fail = drop, however cheap.
```

Step ⑥ can veto steps ①–⑤. Cheap plus bad management is not a discount; it's a
trap with a countdown.

---

## The rule that isn't in the code

The screener exists to discipline its operator, not to decide for them. Both
directions of failure sit with the operator:

- **Confusing a chart with a forecast.** Cumulative past gains are not evidence
  of future doubling. The only honest translation of a target price is
  `fair value ÷ price − 1`, expressed as a percentage.
- **Confusing a gate with a suggestion.** A score passing doesn't mean buy, and
  "it should go up" doesn't reopen a gate that's closed.

So the tool has to be able to produce a *no*, and the operator has to be able to
take it. That is the real purpose of backtesting the tiers: not to prove the
ranking right, but to find where it was wrong.

---

## What this repository is not

- **No positions, no portfolio weights, no personal returns.** The framework is
  public; the book is not.
- **Not investment advice.** See the disclaimer in the [README](README.md).

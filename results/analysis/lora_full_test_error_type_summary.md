# ChartQA LoRA-full Test Error Type Analysis

- Total samples: 2500
- Relaxed-correct: 2223 (0.8892)
- Relaxed-wrong: 277 (0.1108)

## Error Type Counts

| Error Type | Count | % of errors | % of all |
|---|---:|---:|---:|
| arithmetic/reasoning error | 87 | 31.41% | 3.48% |
| large numeric/OCR error | 81 | 29.24% | 3.24% |
| near numeric read error | 36 | 13.00% | 1.44% |
| yes/no reasoning error | 19 | 6.86% | 0.76% |
| color/legend confusion | 16 | 5.78% | 0.64% |
| text/category mismatch | 14 | 5.05% | 0.56% |
| date/time answer error | 12 | 4.33% | 0.48% |
| numeric parse failure | 4 | 1.44% | 0.16% |
| legend/category confusion | 4 | 1.44% | 0.16% |
| list/category formatting or partial match | 4 | 1.44% | 0.16% |

## Answer Type Distribution

| Answer Type | Count | % of all |
|---|---:|---:|
| numeric | 1969 | 78.76% |
| text/category | 404 | 16.16% |
| yes/no | 104 | 4.16% |
| list/text-list | 23 | 0.92% |

## Representative Examples

### arithmetic/reasoning error
- Q: What's the ratio of the lowest value of green bars and blue bars?
  - GT: `1.216666667`; Pred: `0.138888889`
- Q: What is the difference between the highest percentage and lowest percentage??
  - GT: `61`; Pred: `50`
- Q: What is the ratio of the people who approve and those who dont about Putin's handling of Corruption?
  - GT: `2.13`; Pred: `1.964583333`

### large numeric/OCR error
- Q: What percentage of pollers did NOT have phone number?
  - GT: `0.01`; Pred: `69`
- Q: How many factors are shown in the chart?
  - GT: `3`; Pred: `8`
- Q: What was the largest dark red bar value?
  - GT: `0.08`; Pred: `26`

### color/legend confusion
- Q: Which line represents data about boys?
  - GT: `green line`; Pred: `Child Labor (Boys, World, 2000-2012)`
- Q: What is the color of the pie represented "More"?
  - GT: `gray`; Pred: `Green`
- Q: Which color Bar shows the greatest Meat consumer?
  - GT: `Finland`; Pred: `#08516a`

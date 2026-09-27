# ML Challenge 2026: Business Entity Resolution Solution Template

****Team Name:**** [ScriptKnights]

****Team Members:**** [Aman Singh,Akash Babu,Aradhya Gupta,Nitin Kothiyal]

****Submission Date:**** 27 September 2026

**---**

## 1. Executive Summary**

*We developed a blocking-based business entity resolution pipeline to identify matching business records across three noisy data sources. The solution combines multiple complementary blocking strategies with character/token-based name and address similarity features, followed by a HistGradientBoosting classifier to identify true entity matches while supporting multiple matches per reference entity.

The approach focuses on reducing the search space efficiently while preserving high candidate recall, followed by precision-oriented classification optimized for the challenge's F0.5 evaluation metric.*

**---**

## 2. Methodology**

### 2.1 Problem Analysis**

*The dataset contains three sources of business records. Source 1 acts as the reference dataset, while Sources 2 and 3 contain noisy records referring to the same real-world businesses.

During exploratory analysis, we identified several sources of variation:

Business names contain spelling errors, abbreviations, punctuation differences and legal suffix variations.

Names may use different word ordering or representations such as & versus and.

Legal and organizational terms such as Ltd, LLC, Inc, Corporation, Pvt, etc. frequently appear and provide limited matching information.

Addresses contain abbreviations such as Road/Rd, Street/St, Avenue/Ave, etc.

Address components may be missing, reordered or represented using landmarks.

Some records have missing business names or addresses.

The same Source 1 entity can correspond to multiple records in Sources 2 and 3.

Country information was highly useful for restricting comparisons, while the actual country values were treated dynamically rather than hard-coded because the test data contains countries not present in the training data.

Analysis of the ground truth showed that multiple matches are common, so the solution was designed as multi-match entity resolution rather than top-1 matching.*

**### 2.2 Solution Strategy**

*Approach Type: Blocking + Pairwise Feature Engineering + Gradient Boosting Classifier

The solution follows this pipeline:

Raw business records

```
    ↓
```

Text normalization

```
    ↓
```

Multi-channel blocking

```
    ↓
```

Candidate pairs

```
    ↓
```

Pairwise similarity features

```
    ↓
```

HistGradientBoosting classifier

```
    ↓
```

Precision-oriented threshold

```
    ↓
```

Multiple matched entities

```
    ↓
```

Final submission files

The main technical contribution is the use of multiple complementary blocking channels rather than relying on a single exact or fuzzy matching rule. This allows the system to recover matches affected by different types of noise while keeping the candidate space manageable.*

****Approach Type:**** Blocking + Classifier

****Core Innovation:**** Multi-channel blocking using normalized names, meaningful address tokens, name-token pairs and address-token pairs, followed by feature-based gradient boosting classification.

**---**

## 3. Candidate Generation (Blocking)**

*Candidate generation was used to avoid comparing every Source 1 record against every Source 2 and Source 3 record.

The blocking system uses four complementary channels.*

- ****Blocking keys used:**** Country + exact normalized business name, country + meaningful address token, country + meaningful business-name token pair, and country + meaningful address-token pair.

- ****Candidate pairs generated:**** 15,956,701 candidate pairs in the 10,000-entity development experiment.

- ****How you ensured true matches were not lost:**** Multiple blocking channels were evaluated and combined using a union. This allowed a true match to be recovered through either name-based or address-based evidence. The final development blocker recovered 33,459 of 34,567 true positive pairs, giving 96.79% pair recall.

**---**

## 4. Matching Model**

**Features used:**

- Name features: Exact normalized name match, character-level similarity, token-based similarity, partial similarity, relative name length, shared meaningful name tokens and strong similarity indicators.

- Address features: Exact normalized address match, character-level similarity, token-based similarity, partial similarity, relative address length, shared meaningful address tokens and strong similarity indicators.

- Other: Country match, shared numerical components, multiple shared numbers, combined name/address similarity features, strong evidence combinations and source indicator.

****Model type:**** HistGradientBoostingClassifier

****Threshold selection method:**** F0.5 optimization using grouped validation, with Source 1 entities kept together to avoid leakage between training and validation. The final threshold was 0.30.

**---**

## 5. Results & Error Analysis**

- ****F_0.5 Score (macro):**** 0.834725 on grouped development validation; the submitted team solution achieved a leaderboard score of 0.934378.

- ****Common false positives (wrong merges):**** Businesses with similar names or shared generic address components were the main difficult false-positive cases. The classifier combines multiple name, address and numerical signals rather than relying on a single similarity score.

- ****Common false negatives (missed matches):**** Difficult cases included heavily modified names, incomplete addresses, transliteration differences and records with very little overlapping information.

**---**

## 6. Conclusion**

*The solution combines multi-channel blocking with pairwise name and address features and a gradient boosting classifier to efficiently identify multiple matches for each reference business.

The submitted solution achieved a leaderboard score of 0.934378. The development process showed that effective candidate generation is a critical part of large-scale entity resolution because the final classifier can only evaluate candidates produced by the blocking stage.*

**---**

## Appendix

### A. Code Artefacts**

*Your complete, runnable code ships in the submission zip under

`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and

`requirements.txt`). The source contains the candidate generation, feature engineering, model training, test inference and validation components required to reproduce

`output/matching_results.tsv` and `output/candidate_pairs.tsv`.*

### B. Additional Results

*For the 10,000-entity development sample, the final blocking system generated 15,956,701 candidate pairs and recovered 33,459 of 34,567 true positive pairs, resulting in 96.79% pair recall. Multiple blocking strategies were evaluated during development to balance candidate recall and candidate volume.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.

# Feature-evolution confirmatory findings

Values are paired across five seeds and remain provisional until the full 30-run suite is audited and aggregated.

## Balanced 24-to-25 feature scenario

The mapper materially improved latent alignment: stable transferred-prototype cosine similarity increased from 0.6194 without mapping to 0.7666 with mapping (no-mapper minus mapper difference -0.1472; 95% CI [-0.1683, -0.1261]; paired dz -8.665). Euclidean prototype distance changed little, illustrating why both magnitude-sensitive and angular alignment diagnostics are retained.

Predictive effects were modest and uncertain with five seeds. Mapper transition accuracy was 0.5072 versus 0.4976 without it (mapper advantage 0.0096; paired interval for no-mapper minus mapper [-0.0247, 0.0055]). The mapper recovered 25.8 observations sooner on average, but the interval crossed zero. Early-recovery, stable accuracy, stable G-Mean, and stable macro PR-AUC were essentially unchanged.

Thus, the balanced scenario confirms that the mapper changes latent alignment as intended, but does not establish a stable predictive advantage. Any transfer claim should focus on alignment and possible transition/recovery effects rather than final accuracy.

## Stream 2 expansion: 21-to-28 features

The expansion scenario again confirms functional latent alignment. With the mapper, stable transferred-prototype cosine similarity was 0.7517 versus 0.5609 without it (no-mapper minus mapper difference -0.1908; 95% CI [-0.2215, -0.1601]). Stable Euclidean distance was 3.1571 with mapping and 3.2722 without it (difference +0.1151; 95% CI [0.0268, 0.2035]). Both diagnostics therefore favor the mapper.

Predictive outcomes did not: transition, early-recovery, and stable accuracy differences were all below 0.005 in magnitude, stable G-Mean and macro PR-AUC were nearly identical, and recovery-time intervals crossed zero. The mapper successfully aligns the spaces but offers no demonstrated predictive gain when Stream 2 expands in this held-out region.

## Stream 2 contraction: 27-to-22 features

The contraction scenario repeats the angular-alignment result: stable cosine similarity was 0.7280 with mapping versus 0.5322 without it (no-mapper minus mapper -0.1959; 95% CI [-0.2457, -0.1461]). Euclidean distance favored mapping in its point estimate but had an interval crossing zero.

The mapper again did not improve predictive outcomes. Transition and early-recovery accuracy differences were small and uncertain; no-mapper point estimates were slightly higher for stable accuracy and G-Mean. Stable macro PR-AUC was higher without mapping by 0.00734 (95% CI [0.00107, 0.01362]). Recovery-time uncertainty was wide.

Across balanced, expansion, and contraction scenarios, unequal-dimensional execution and latent angular alignment are consistently validated. Stable predictive improvement is not: the mapper is neutral in balanced/expansion and modestly worse for stable probability ranking under contraction. This is the defensible answer to the feature-transfer question under the held-out protocol.

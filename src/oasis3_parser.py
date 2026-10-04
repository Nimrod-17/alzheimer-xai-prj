import re
from pathlib import Path

import nibabel as nib
import pandas as pd

from src.interfaces import DataLoaderInterface, LabelingStrategyInterface, ScanSelectorInterface


class BIDSFilenameParser:
    '''
    Extract BIDS entities (subject, session day, acq, echo, run) from an OASIS-3 T1w filename.
    Handles both the 'ses-' and the 'sess-' spellings found in the NITRC exports.
    '''
    PATTERN = re.compile(
        r'^sub-(?P<subject>OAS3\d{4})'
        r'_sess?-d(?P<day>\d+)'
        r'(?:_acq-(?P<acq>[^_]+))?'
        r'(?:_echo-(?P<echo>\d+))?'
        r'(?:_run-(?P<run>\d+))?'
        r'_T1w\.nii(?:\.gz)?$'
    )

    def parse(self, filename: str) -> dict | None:
        match = self.PATTERN.match(filename)
        if not match:
            return None

        entities = match.groupdict()
        return {
            'Subject': entities['subject'],
            'MRIDay': int(entities['day']),
            'Acq': entities['acq'],
            'Echo': int(entities['echo']) if entities['echo'] else None,
            'Run': int(entities['run']) if entities['run'] else 1,
        }


class VolumeQualityFilter:
    '''
    Accept only whole-brain 3D volumes, judged from the NIfTI header alone (no voxel data is read).
    Rejects partial-coverage slabs (e.g. acq-hippocampus), thick-slice 2D acquisitions (e.g. echo-N)
    and files without spatial orientation (qform_code = sform_code = 0, e.g. the 256x256x128 sagittal
    scans): their left/right direction cannot be recovered, which would corrupt registration and XAI maps.
    '''
    def __init__(self, min_extent_mm: float = 150.0, max_voxel_mm: float = 1.5):
        self.min_extent_mm = min_extent_mm
        self.max_voxel_mm = max_voxel_mm

    def is_valid(self, file_path: Path) -> bool:
        try:
            header = nib.load(str(file_path)).header
        except Exception:
            return False

        shape = [s for s in header.get_data_shape() if s > 1]
        if len(shape) != 3:
            return False

        if int(header['qform_code']) == 0 and int(header['sform_code']) == 0:
            return False

        zooms = header.get_zooms()[:3]
        extents = [n * z for n, z in zip(header.get_data_shape()[:3], zooms)]
        return max(zooms) <= self.max_voxel_mm and min(extents) >= self.min_extent_mm


class OASIS3ScanIndexer:
    '''
    Index the flat MRI directory into a DataFrame with one valid T1w scan per MR session.
    When a session has several valid runs, the lowest run number is kept.
    '''
    def __init__(self, mri_dir: str, filename_parser: BIDSFilenameParser, quality_filter: VolumeQualityFilter):
        self.mri_dir = Path(mri_dir)
        self.filename_parser = filename_parser
        self.quality_filter = quality_filter

    def index(self) -> pd.DataFrame:
        records = []
        unparsed, rejected = 0, 0

        for file_path in sorted(self.mri_dir.glob('*.nii*')):
            entities = self.filename_parser.parse(file_path.name)
            if entities is None:
                unparsed += 1
                continue
            if not self.quality_filter.is_valid(file_path):
                rejected += 1
                continue
            records.append({**entities, 'Path': str(file_path.resolve())})

        scans = pd.DataFrame(records)
        print(f"Indexed {len(scans)} valid volumes ({rejected} rejected by quality filter, {unparsed} unparsable names).")

        scans = (
            scans.sort_values(['Subject', 'MRIDay', 'Run'])
                 .drop_duplicates(subset=['Subject', 'MRIDay'], keep='first')
                 .reset_index(drop=True)
        )
        scans['ID'] = scans['Subject'] + '_MR_d' + scans['MRIDay'].astype(str).str.zfill(4)
        print(f"Kept {len(scans)} MR sessions from {scans['Subject'].nunique()} subjects.")
        return scans


class OASIS3ClinicalParser:
    '''
    Read the UDS B4 form (CDR global staging) into one row per clinical visit.
    '''
    def __init__(self, cdr_csv_path: str):
        self.cdr_csv_path = cdr_csv_path

    def parse(self) -> pd.DataFrame:
        df = pd.read_csv(self.cdr_csv_path)
        df = df.dropna(subset=['CDRTOT'])

        visits = pd.DataFrame({
            'Subject': df['OASISID'],
            'VisitDay': df['days_to_visit'].astype(int),
            'CDR': df['CDRTOT'].astype(float),
            'Dx': df['dx1'].fillna('.').astype(str).str.strip(),
            'MMSE': df['MMSE'],
        })
        print(f"Loaded {len(visits)} clinical visits from {visits['Subject'].nunique()} subjects.")
        return visits


class NearestVisitMatcher:
    '''
    Pair each MR session with the temporally closest clinical visit of the same subject.
    Sessions without a visit within max_gap_days are discarded.
    '''
    def __init__(self, max_gap_days: int = 180):
        self.max_gap_days = max_gap_days

    def match(self, scans: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        matched = pd.merge_asof(
            scans.sort_values('MRIDay'),
            visits.sort_values('VisitDay'),
            left_on='MRIDay',
            right_on='VisitDay',
            by='Subject',
            direction='nearest',
            tolerance=self.max_gap_days,
        )
        matched = matched.dropna(subset=['CDR']).copy()
        matched['VisitDay'] = matched['VisitDay'].astype(int)
        matched['DayGap'] = (matched['MRIDay'] - matched['VisitDay']).abs()

        print(f"Matched {len(matched)}/{len(scans)} sessions to a clinical visit within {self.max_gap_days} days.")
        return matched.sort_values(['Subject', 'MRIDay']).reset_index(drop=True)


class ADDiagnosisRule:
    '''
    Decide whether a clinical visit corresponds to Alzheimer-type impairment:
    CDR > 0 with an AD primary diagnosis (UDS "AD Dementia", "AD dem ...", "DAT").
    '''
    AD_PATTERN = r'^(?:AD Dementia|AD dem|DAT)'
    EXCLUDED_PATTERN = r'cannot be primary'

    def is_ad(self, df: pd.DataFrame) -> pd.Series:
        return (
            (df['CDR'] > 0)
            & df['Dx'].str.contains(self.AD_PATTERN, regex=True)
            & ~df['Dx'].str.contains(self.EXCLUDED_PATTERN, regex=True)
        )


class CDRBinaryLabeler(LabelingStrategyInterface):
    '''
    Label 0 for CDR = 0 (cognitively normal), label 1 for CDR > 0 (any cognitive impairment).
    '''
    def label(self, scans: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        scans = scans.copy()
        scans['Label'] = (scans['CDR'] > 0).astype(int)
        return scans


class ADvsCNLabeler(LabelingStrategyInterface):
    '''
    Cross-sectional labelling based only on the matched visit:
    label 0 = CDR 0 with a "Cognitively normal" diagnosis, label 1 = Alzheimer-type impairment.
    Everything else (non-AD dementias, uncertain diagnoses) is dropped.
    '''
    def __init__(self, ad_rule: ADDiagnosisRule):
        self.ad_rule = ad_rule

    def label(self, scans: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        is_cn = (scans['CDR'] == 0) & (scans['Dx'] == 'Cognitively normal')
        is_ad = self.ad_rule.is_ad(scans)

        keep = is_cn | is_ad
        scans = scans[keep].copy()
        scans['Label'] = is_ad[keep].astype(int)
        return scans


class ADRiskLabeler(LabelingStrategyInterface):
    '''
    Prognostic labelling that uses each subject's clinical follow-up.

    Label 1 ("AD or progressing to AD"):
      - the matched visit already shows Alzheimer-type impairment, or
      - the subject receives a first AD diagnosis within `conversion_horizon_days` after the scan.
    Label 0 ("stable cognitively normal"):
      - CDR 0 and "Cognitively normal" at the scan, no visit with CDR > 0 afterwards,
        and clinical follow-up lasting at least `min_followup_days` after the scan.
    Dropped (ambiguous):
      - non-AD impairment, conversions beyond the horizon, normal subjects with too short a
        follow-up, and scans of subjects already diagnosed with AD but not at the matched visit.
    '''
    def __init__(self, ad_rule: ADDiagnosisRule, conversion_horizon_days: int = 1825, min_followup_days: int = 1095):
        self.ad_rule = ad_rule
        self.conversion_horizon_days = conversion_horizon_days
        self.min_followup_days = min_followup_days

    def label(self, scans: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        history = self._summarize_history(scans, visits)
        scans = scans.merge(history, on=['Subject', 'MRIDay'], how='left')

        ad_now = self.ad_rule.is_ad(scans)
        ad_before = scans['FirstADDay'] <= scans['VisitDay']
        days_to_ad = scans['FirstADDay'] - scans['MRIDay']
        converts = ~ad_before & (days_to_ad <= self.conversion_horizon_days)

        stable_cn = (
            (scans['CDR'] == 0)
            & (scans['Dx'] == 'Cognitively normal')
            & ~scans['ImpairedAfter']
            & (scans['FollowUpDays'] >= self.min_followup_days)
        )

        positive = ad_now | converts
        negative = stable_cn & ~positive & ~ad_before

        keep = positive | negative
        scans = scans[keep].copy()
        scans['Label'] = positive[keep].astype(int)
        scans['DaysToAD'] = days_to_ad[keep]
        return scans.drop(columns=['FirstADDay', 'ImpairedAfter', 'FollowUpDays'])

    def _summarize_history(self, scans: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        '''
        For every (subject, MRI day) pair, summarize what the clinical record says about the future.
        '''
        visits = visits.assign(IsAD=self.ad_rule.is_ad(visits))
        first_ad_day = visits[visits['IsAD']].groupby('Subject')['VisitDay'].min().rename('FirstADDay')

        rows = []
        for (subject, mri_day, visit_day), _ in scans.groupby(['Subject', 'MRIDay', 'VisitDay']):
            subject_visits = visits[visits['Subject'] == subject]
            later = subject_visits[subject_visits['VisitDay'] > visit_day]
            rows.append({
                'Subject': subject,
                'MRIDay': mri_day,
                'ImpairedAfter': bool((later['CDR'] > 0).any()),
                'FollowUpDays': subject_visits['VisitDay'].max() - mri_day,
            })

        return pd.DataFrame(rows).merge(first_ad_day, on='Subject', how='left')


class ClosestVisitPerSubjectSelector(ScanSelectorInterface):
    '''
    Keep a single scan per subject: the one with the tightest MRI-to-clinic alignment.
    Ties are broken by the earliest session.
    '''
    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.sort_values(['Subject', 'DayGap', 'MRIDay'])
              .drop_duplicates(subset=['Subject'], keep='first')
              .reset_index(drop=True)
        )


class AllScansSelector(ScanSelectorInterface):
    '''
    Keep every labelled scan. Only safe together with a subject-level train/val/test split.
    '''
    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.reset_index(drop=True)


class OASIS3DataLoader(DataLoaderInterface):
    '''
    Compose indexing, clinical parsing, temporal matching, labelling and scan selection.
    Output keeps the 'ID', 'Path' and 'Label' columns expected by MRIDataset,
    plus 'Subject' for subject-level splitting and clinical columns for analysis.
    '''
    OUTPUT_COLUMNS = ['ID', 'Subject', 'Path', 'Label', 'CDR', 'Dx', 'MMSE', 'MRIDay', 'VisitDay', 'DayGap']

    # Optional columns produced only by some labelling strategies
    OPTIONAL_COLUMNS = ['DaysToAD']

    def __init__(
        self,
        indexer: OASIS3ScanIndexer,
        clinical_parser: OASIS3ClinicalParser,
        matcher: NearestVisitMatcher,
        labeler: LabelingStrategyInterface,
        selector: ScanSelectorInterface,
    ):
        self.indexer = indexer
        self.clinical_parser = clinical_parser
        self.matcher = matcher
        self.labeler = labeler
        self.selector = selector

    def load_data(self) -> pd.DataFrame:
        scans = self.indexer.index()
        visits = self.clinical_parser.parse()

        matched = self.matcher.match(scans, visits)
        labelled = self.labeler.label(matched, visits)
        selected = self.selector.select(labelled)

        counts = selected['Label'].value_counts().sort_index().to_dict()
        print(f"Final dataset: {len(selected)} scans from {selected['Subject'].nunique()} subjects | label counts: {counts}")
        extra = [c for c in self.OPTIONAL_COLUMNS if c in selected.columns]
        return selected[self.OUTPUT_COLUMNS + extra]

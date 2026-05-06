
"""
ii_embeddings/dhiv_embed_test
Dhivehi intrinsic embedding evaluation suite:
  1. Morphological analogies (5,818 analogies across 22 categories)
  2. Outlier detection (does the model spot the odd word out?)
  3. Word categorization purity (k-means on category exemplars)
Each test is independent. Overall summary printed at the end.
"""
import argparse
import os
import sys
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans

# 1. MORPHOLOGY DATA
plural_nonhuman = [
    ("ގެ",         "ގެތައް"),
    ("ފޮތް",       "ފޮތްތައް"),
    ("މައްސަލަ",   "މައްސަލަތައް"),
    ("ހައްގު",     "ހައްގުތައް"),
    ("ބަސް",       "ބަސްތައް"),
    ("ދަތުރު",     "ދަތުރުތައް"),
    ("ޖަލްސާ",    "ޖަލްސާތައް"),
    ("ސަރުކާރު",   "ސަރުކާރުތައް"),
    ("މަގު",       "މަގުތައް"),
    ("ކަން",       "ކަންތައް"),
    ("ފިހާރަ",    "ފިހާރަތައް"),
    ("ރަށް",       "ރަށްތައް"),
    ("ވިލާތް",    "ވިލާތްތައް"),
    ("މިސްކިތް",  "މިސްކިތްތައް"),
    ("އޮފީސް",    "އޮފީސްތައް"),
    ("ބިން",       "ބިންތައް"),
    ("ފެން",       "ފެންތައް"),
    ("ވަގުތު",    "ވަގުތުތައް"),
    ("ތަން",       "ތަންތައް"),
    ("ބަދަލު",    "ބަދަލުތައް"),
]
plural_human = [
    ("ކުއްޖާ",       "ކުދިން"),
    ("މީހާ",         "މީހުން"),
    ("ބައްޕަ",       "ބައްޕަމެން"),
    ("މަންމަ",       "މަންމަމެން"),
    ("ކޮއްކޮ",      "ކޮއްކޮމެން"),
    ("ރައީސް",       "ރައީސުން"),
    ("ފުލުސް",       "ފުލުހުން"),
    ("ޒުވާނާ",      "ޒުވާނުން"),
    ("ބޭބެ",         "ބެއިން"),
    ("ވެރި",         "ވެރިން"),
    ("އެހީތެރިޔާ",   "އެހީތެރިން"),
    ("އިލްމުވެރިޔާ", "އިލްމުވެރިން"),
    ("ރަށުވެރިޔާ",   "ރަށުވެރިން"),
    ("މުދައްރިސު",   "މުދައްރިސުން"),
    ("ޑޮކްޓަރު",    "ޑޮކްޓަރުން"),
    ("ވަޒީރު",       "ވަޒީރުން"),
    ("ދަރިވަރު",   "ދަރިވަރުން"),
    ("ކުޅުންތެރިޔާ", "ކުޅުންތެރިން"),
]
plural_associative = [
    ("މަންމަ",       "މަންމަމެން"),
    ("ބައްޕަ",       "ބައްޕަމެން"),
    ("ކޮއްކޮ",      "ކޮއްކޮމެން"),
    ("ކަލޭ",         "ކަލޭމެން"),
    ("އަޅުގަނޑު",    "އަޅުގަނޑުމެން"),
    ("އަހަރެން",     "އަހަރެމެން"),
    ("ތިމަންނަ",    "ތިމަންނަމެން"),
    ("ކާފަ",         "ކާފަމެން"),
    ("މާމަ",         "މާމަމެން"),
    ("ބޭބެ",         "ބޭބެމެން"),
    ("ރައީސް",       "ރައީސުން"),
    ("ވެރި",         "ވެރިން"),
    ("މުދައްރިސު",   "މުދައްރިސުން"),
    ("ޓީޗަރު",       "ޓީޗަރުން"),
    ("ޑޮކްޓަރު",    "ޑޮކްޓަރުން"),
    ("ކުއްޖާ",       "ކުދިން"),
    ("މީހާ",         "މީހުން"),
]
kinship_gender = [
    ("ފިރިހެން",    "އަންހެން"),
    ("ބައްޕަ",       "މަންމަ"),
    ("ދޮންބައްޕަ",  "ދޮންމަންމަ"),
    ("ބަފައިބެ",    "މައިދައިތަ"),
    ("ޅިޔަނު",       "ފަހަރި"),
    ("ކާފަ",         "މާމަ"),
    ("މުނިކާފަ",    "މުނިމާމަ"),
    ("ހޫރުކާފަ",    "ހޫރުމާމަ"),
    ("ބޭބެ",         "ދައްތަ"),
    ("ފިރިމީހާ",    "އަނބިމީހާ"),
    ("ދަރިކަލުން",  "ދަރިކަނބަލުން"),
    ("ރަސްގެފާނު",  "ރާނީ"),
    ("ބޮޑުބޭބެ",   "ބޮޑުދައިތަ"),
    ("އަމީރު",       "އަމީރާ"),
    ("އެބޭކަލުން",  "އެބޭކަނބަލުން"),
    ("އެބޭފުޅާ",    "އެކަމަނާ"),
]
non_human_cases = [
    ("ބަސް",        "language",   "ބަހުގެ",       "ބަހަށް",       "ބަހުގައި",      "ބަހުން",        "ބަހާ"),
    ("ފޮތް",        "book",       "ފޮތުގެ",       "ފޮތަށް",       "ފޮތުގައި",      "ފޮތުން",        "ފޮތާ"),
    ("ފޮތްތައް",    "books",      "ފޮތްތަކުގެ",   "ފޮތްތަކަށް",   "ފޮތްތަކުގައި",  "ފޮތްތަކުން",    "ފޮތްތަކާ"),
    ("ގައުމު",      "nation",     "ގައުމުގެ",     "ގައުމަށް",     "ގައުމުގައި",    "ގައުމުން",      "ގައުމާ"),
    ("ދީން",        "religion",   "ދީނުގެ",       "ދީނަށް",       "ދީނުގައި",      "ދީނުން",        "ދީނާ"),
    ("ކުށް",        "crime",      "ކުށުގެ",       "ކުށަށް",       "ކުށުގައި",      "ކުށުން",        "ކުށާ"),
    ("ދަތުރު",      "journey",    "ދަތުރުގެ",     "ދަތުރަށް",     "ދަތުރުގައި",    "ދަތުރުން",      "ދަތުރާ"),
    ("ޖަލްސާ",     "assembly",   "ޖަލްސާގެ",     "ޖަލްސާއަށް",   "ޖަލްސާގައި",    "ޖަލްސާއިން",    "ޖަލްސާއާ"),
    ("ރާއްޖެ",      "maldives",   "ރާއްޖޭގެ",     "ރާއްޖެއަށް",   "ރާއްޖޭގައި",    "ރާއްޖެއިން",    "ރާއްޖެއާ"),
    ("ތަކެތި",      "things",     "ތަކެތީގެ",     "ތަކެއްޗަށް",   "ތަކެތީގައި",    "ތަކެތިން",      "ތަކެއްޗާ"),
    ("ބޯ",          "head",       "ބޮލުގެ",       "ބޮލަށް",       "ބޮލުގައި",      "ބޮލުން",        "ބޮލާ"),
    ("މައްސަލަ",   "case",       "މައްސަލައިގެ", "މައްސަލައަށް", "މައްސަލައިގައި","މައްސަލައިން",  "މައްސަލައާ"),
    ("ފޮށި",        "box",        "ފޮށީގެ",       "ފޮށްޓަށް",     "ފޮށީގައި",      "ފޮށިން",        "ފޮށްޓާ"),
    ("މާލެ",        "Male",       "މާލޭގެ",       "މާލެއަށް",     "މާލޭގައި",      "މާލެއިން",      "މާލެއާ"),
    ("ތަން",        "place",      "ތަނުގެ",       "ތަނަށް",       "ތަނުގައި",      "ތަނުން",        "ތަނާ"),
    ("ވަގުތު",     "time",       "ވަގުތުގެ",     "ވަގުތަށް",     "ވަގުތުގައި",    "ވަގުތުން",      "ވަގުތާ"),
    ("ސަބަބު",     "reason",     "ސަބަބުގެ",     "ސަބަބަށް",     "ސަބަބުގައި",    "ސަބަބުން",      "ސަބަބާ"),
    ("ގޮތް",        "manner",     "ގޮތުގެ",       "ގޮތަށް",       "ގޮތުގައި",      "ގޮތުން",        "ގޮތާ"),
    ("ބަދަލު",      "change",     "ބަދަލުގެ",     "ބަދަލަށް",     "ބަދަލުގައި",    "ބަދަލުން",      "ބަދަލާ"),
    ("ގެ",          "house",      "ގޭގެ",         "ގެއަށް",       "ގޭގައި",        "ގެއިން",        "ގޭއާ"),
    ("ހައްގުތައް",  "rights",     "ހައްގުތަކުގެ", "ހައްގުތަކަށް", "ހައްގުތަކުގައި","ހައްގުތަކުން",  "ހައްގުތަކާ"),
    ("މަސައްކަތް",  "work",       "މަސައްކަތުގެ", "މަސައްކަތަށް", "މަސައްކަތުގައި","މަސައްކަތުން",  "މަސައްކަތާ"),
]
human_cases = [
    ("ކޮއްކޮ",   "younger sibling", "ކޮއްކޮގެ",    "ކޮއްކޮއަށް",    "ކޮއްކޮއާ"),
    ("ކުދިން",   "children",        "ކުދިންގެ",    "ކުދިންނަށް",    "ކުދިންނާ"),
    ("ފުލުހުން", "the police",      "ފުލުހުންގެ",  "ފުލުހުންނަށް",  "ފުލުހުންނާ"),
    ("މީހެއް",  "a person",        "މީހެއްގެ",    "މީހަކަށް",      "މީހަކާ"),
    ("ބައްޕަ",  "father",          "ބައްޕަގެ",    "ބައްޕައަށް",    "ބައްޕައާ"),
    ("އޭނަ",    "he/she",          "އޭނަގެ",      "އޭނައަށް",     "އޭނައާ"),
    ("އަހަރެން", "I",               "އަހަރެންގެ",  "އަހަންނަށް",   "އަހަންނާ"),
]
finite_indicative = [
    ("ހަދަނީ",   "ހަދާ",   "ހެދި",    "ހެދީ",    "ހަދާނެ"),
    ("ކިޔަނީ",   "ކިޔާ",   "ކިއި",    "ކިއީ",    "ކިޔާނެ"),
    ("ލިޔަނީ",   "ލިޔާ",   "ލިޔުނު",  "ލިޔުނީ",  "ލިޔާނެ"),
    ("ހިނގަނީ",  "ހިނގާ",  "ހިނގި",   "ހިނގީ",   "ހިނގާނެ"),
    ("ބަލަނީ",   "ބަލާ",   "ބެލި",    "ބެލީ",    "ބަލާނެ"),
    ("ކުޅެނީ",   "ކުޅޭ",   "ކުޅުނު",  "ކުޅުނީ",  "ކުޅޭނެ"),
    ("ފެށެނީ",   "ފެށޭ",   "ފެށުނު",  "ފެށުނީ",  "ފެށޭނެ"),
    ("ކުރަނީ",   "ކުރޭ",   "ކުރި",    "ކުރީ",    "ކުރާނެ"),
    ("ވަނީ",     "ވޭ",     "ވި",      "ވީ",      "ވާނެ"),
    ("ދެނީ",     "ދޭ",     "ދިން",    "ދިނީ",    "ދޭނެ"),
    ("ދަނީ",     "ދޭ",     "ދިޔަ",    "ދިޔައީ",  "ދާނެ"),
    ("ކަނީ",     "ކައި",   "ކޭ",      "ކެއީ",    "ކާނެ"),
    ("އަންނަނީ", "އާދޭ",   "އައި",    "އައީ",    "އަންނާނެ"),
    ("ހުންނަނީ", "ހުރޭ",   "ހުރި",    "ހުރީ",    "ހުންނާނެ"),
    ("އިންނަނީ", "އިނދޭ",  "އިން",    "އިނީ",    "އިންނާނެ"),
    ("އޮންނަނީ", "އޮވޭ",   "އޮތި",    "އޮތީ",    "އޮންނާނެ"),
]
imperatives = [
    ("ހަދަނީ",   "ހަދާ",   "ހަދާތި"),
    ("ކިޔަނީ",   "ކިޔާ",   "ކިޔާތި"),
    ("ލިޔަނީ",   "ލިޔާ",   "ލިޔާތި"),
    ("ހިނގަނީ",  "ހިނގާ",  "ހިނގާތި"),
    ("ބަލަނީ",   "ބަލާ",   "ބަލާތި"),
    ("ކުޅެނީ",   "ކުޅޭ",   "ކުޅޭތި"),
    ("ކުރަނީ",   "ކުރޭ",   "ކުރާތި"),
    ("ވަނީ",     "ވޭ",     "ވާތި"),
    ("އަންނަނީ", "އާދޭ",   "އަންނާތި"),
    ("ހުންނަނީ", "ހުރޭ",   "ހުންނާތި"),
    ("އިންނަނީ", "އިނދޭ",  "އިންނާތި"),
    ("އޮންނަނީ", "އޮވޭ",   "އޮންނާތި"),
]
perfect = [
    ("ހަދަނީ",   "ހަދައިފި"),
    ("ކިޔަނީ",   "ކިޔައިފި"),
    ("ލިޔަނީ",   "ލިޔެފި"),
    ("ހިނގަނީ",  "ހިނގައިފި"),
    ("ބަލަނީ",   "ބަލައިފި"),
    ("ކުޅެނީ",   "ކުޅެފި"),
    ("ފެށެނީ",   "ފެށިއްޖެ"),
    ("ކުރަނީ",   "ކޮށްފި"),
    ("ވަނީ",     "ވެއްޖެ"),
    ("ދެނީ",     "ދީފި"),
    ("ދަނީ",     "ގޮސްފި"),
    ("ކަނީ",     "ކައިފި"),
    ("އަންނަނީ", "އައިސްފި"),
    ("ހުންނަނީ", "ހުރެއްޖެ"),
    ("އިންނަނީ", "އިންދެފި"),
    ("އޮންނަނީ", "އޮވެއްޖެ"),
]
potential = [
    ("ހަދަނީ",   "ހަދާފާނެ"),
    ("ކިޔަނީ",   "ކިޔާފާނެ"),
    ("ލިޔަނީ",   "ލިޔާފާނެ"),
    ("ހިނގަނީ",  "ހިނގާފާނެ"),
    ("ބަލަނީ",   "ބަލާފާނެ"),
    ("ކުޅެނީ",   "ކުޅެފާނެ"),
    ("ފެށެނީ",   "ފެށިދާނެ"),
    ("ކުރަނީ",   "ކޮށްފާނެ"),
    ("ވަނީ",     "ވެދާނެ"),
    ("ދެނީ",     "ދީފާނެ"),
    ("ދަނީ",     "ދެވިދާނެ"),
    ("ކަނީ",     "ކެވިދާނެ"),
    ("އަންނަނީ", "އައިސްފާނެ"),
    ("ހުންނަނީ", "ހުރެދާނެ"),
    ("އިންނަނީ", "އިނދެދާނެ"),
    ("އޮންނަނީ", "އޮވެދާނެ"),
]
converb = [
    ("ހަދަނީ",   "ހަދައި"),
    ("ކިޔަނީ",   "ކިޔައި"),
    ("ލިޔަނީ",   "ލިޔެ"),
    ("ހިނގަނީ",  "ހިނގައި"),
    ("ބަލަނީ",   "ބަލައި"),
    ("ކުޅެނީ",   "ކުޅެ"),
    ("ފެށެނީ",   "ފެށި"),
    ("ކުރަނީ",   "ކޮށް"),
    ("ވަނީ",     "ވެ"),
    ("ދެނީ",     "ދީ"),
    ("ދަނީ",     "ގޮސް"),
    ("ކަނީ",     "ކައި"),
    ("އަންނަނީ", "އައިސް"),
    ("ހުންނަނީ", "ހުރެ"),
    ("އިންނަނީ", "އިނދެ"),
    ("އޮންނަނީ", "އޮވެ"),
]
honorific = [
    ("ހަދަނީ",   "ހައްދަވަނީ"),
    ("ކިޔަނީ",   "ކިޔުއްވަނީ"),
    ("ލިޔަނީ",   "ލިޔުއްވަނީ"),
    ("ބަލަނީ",   "ބައްލަވަނީ"),
    ("ކުޅެނީ",   "ކުޅުއްވަނީ"),
    ("ދެނީ",     "ދެއްވަނީ"),
    ("ކުރަނީ",   "ކުރައްވަނީ"),
    ("ދަނީ",     "ވަޑައިގަންނަވަނީ"),
    ("ބުނަނީ",   "ވިދާޅުވަނީ"),
    ("ގުޅަނީ",   "ގުޅުއްވަނީ"),
    ("ހޯދަނީ",   "ހޯއްދަވަނީ"),
    ("ހިންގަނީ", "ހިންގަވަނީ"),
    ("ނެރެނީ",   "ނެރުއްވަނީ"),
    ("ހޮވަނީ",   "ހޮއްވަވަނީ"),
    ("ހުންނަނީ", "ހުންނަވަނީ"),
    ("އިންނަނީ", "އިންނަވަނީ"),
]

# 2. OUTLIER DETECTION DATA

dhivehi_outlier_sets = [
    # Family terms
    (['ދައްތަ', 'ބައްޕަ', 'މަންމަ', 'ކޮއްކޮ', 'ގެ'], 'ގެ'),
    (['އަންހެނުން', 'ފިރިމީހާ', 'އަނބިމީހާ', 'ދަރިފުޅު', 'ހިނގުން'], 'ހިނގުން'),
    (['ކާފަ', 'މާމަ', 'ބޮޑުދައިތަ', 'ބޮޑުބޭބެ', 'ގޮނޑި'], 'ގޮނޑި'),
    # Colors
    (['ކަޅު', 'ރަތް', 'ނޫ', 'ފެހި', 'ބޮޑު'], 'ބޮޑު'),
    (['ހުދު', 'ރީނދޫ', 'އަޅި', 'ދަނބު', 'ދުވުން'], 'ދުވުން'),
    (['ބުޅާ', 'ދިއްލުން', 'އަނދިރި', 'ދޮން', 'ފަނޑު'], 'ބުޅާ'),
    # Animals
    (['މިޔަރު', 'ކަކުނި', 'ބުޅާ', 'އަސް', 'ފޮތް'], 'ފޮތް'),
    (['ކުއްތާ', 'ވެލާ', 'މިޔަރު', 'މަސް', 'ބޯވަ'], 'ކުއްތާ'),
    (['ވަކި', 'ކެހެރި', 'ފައި', 'ފިޔަ', 'ކަދުރު'], 'ކަދުރު'),
    # Numbers
    (['ފަނަރަ', 'އެކެއް', 'ސަތޭކަ', 'ތިން', 'ކާކު'], 'ކާކު'),
    (['އަވި', 'ފަންސާސް', 'ސާޅީސް', 'ތިރީސް', 'ވިހި'], 'އަވި'),
    (['މިލިއަން', 'ހާސް', 'ސަތޭކަ', 'ލައްކަ', 'ފިނި'], 'ފިނި'),
    # Body parts (gold = މަސް since ޕާން not in input list)
    (['މަސް', 'އަތް', 'ކަކޫ', 'ލޯ', 'ސިކުނޑި'], 'މަސް'),
    (['ފަންސޫރު', 'ދޫ', 'ނޭފަތް', 'ތުންފަތް', 'ކަންފަތް'], 'ފަންސޫރު'),
    (['ގަސް', 'އަތް', 'ނާރު', 'އިނގިލި', 'ނިޔަފަތި'], 'ގަސް'),
    # Days
    (['ހުކުރު', 'ހޯމަ', 'އަންގާރަ', 'ބުދަ', 'މިއަދު'], 'މިއަދު'),
    (['މާދަން', 'ޖުމްހޫރީ', 'މިނިވަން', 'އާށޫރާ', 'އީދު'], 'މާދަން'),
    (['ހެދުން', 'ބުދަ', 'އަންގާރަ', 'ހޯމަ', 'އާދިއްތަ'], 'ހެދުން'),
    # Verbs
    (['ދުވަނީ', 'ދަނީ', 'އަންނަނީ', 'ހިނގަނީ', 'ނިދަނީ'], 'ނިދަނީ'),
    (['ބިންދަނީ', 'ހިނގަނީ', 'ދުވަނީ', 'ފުންމަނީ', 'ފަތަނީ'], 'ބިންދަނީ'),
    # Food
    (['ގުޅަ', 'ދޮންކެޔޮ', 'މަސްހުނި', 'ރޮށި', 'ފެން'], 'ފެން'),
    (['ބަތް', 'ރިހަ', 'ރޮށި', 'ހަވާދު', 'ތަށި'], 'ތަށި'),
    (['ގުޅަ', 'ލުނބޯ', 'ދޮންކެޔޮ', 'ކަރާ', 'ނޭފަތް'], 'ނޭފަތް'),
    # Other
    (['ފަލަ', 'ކާއިނާތު', 'ދުނިޔެ', 'ހަނދު', 'އިރު'], 'ފަލަ'),
    (['ފިނި', 'ގޯތި', 'ކޮޓަރި', 'ގެ', 'އިމާރާތް'], 'ފިނި'),
    (['ހެދުން', 'ސޯޓު', 'ގަމީސް', 'ފައިވާން', 'ދަނޑި'], 'ދަނޑި'),
    (['ވިއްސާރަ', 'ވާރޭ', 'ތޫފާން', 'ގުގުރި', 'ބިއްލޫރި'], 'ބިއްލޫރި'),
    (['އުފާ', 'ހިތާމަ', 'ރުޅި', 'ބިރު', 'ދަބަސް'], 'ދަބަސް'),
    (['ވަރުބަލި', 'ލޯބި', 'ލަދު', 'ފޫހި', 'ކަނޑު'], 'ކަނޑު'),
    (['ފޮތް', 'ގަލަން', 'ފަންސޫރު', 'ލެކްޗަރު', 'ކިރު'], 'ކިރު'),
    (['ދޯނި', 'ލޯންޗު', 'ބޯޓު', 'ފެރީ', 'ހިތްވަރު'], 'ހިތްވަރު'),
    (['ލޮރީ', 'ސައިކަލު', 'ކާރު', 'ބަސް', 'ތެދުވެރިކަން'], 'ތެދުވެރިކަން'),
    (['ގާޒީ', 'ކާތިބު', 'ވަކީލު', 'ވަޒީރު', 'މަސް'], 'މަސް'),
    (['ކުލާސް', 'މުދައްރިސު', 'އިމްތިހާނު', 'ދިރާސާ', 'މީދާ'], 'މީދާ'),
    (['ހާލި', 'އަނބު', 'ކުކުޅު', 'ދޫނި', 'ކާޅު'], 'އަނބު'),
]

# 3. WORD CATEGORIZATION DATA
dhiv_categories = {
    'animals':  ['ބިޗޫ', 'ކުރަފި', 'ދޫނި', 'މަސް', 'ވެލާ',
                 'ކަކުނި', 'މުސަޅު', 'މިޔަރު', 'ކުއްތާ', 'ބުޅާ'],
    'food':     ['ޕާން', 'ބިސް', 'ފިޔާ', 'ކަދުރު', 'ބަތް',
                 'ރޮށި', 'މަސްހުނި', 'އާފަލު', 'ދޮންކެޔޮ', 'ގުޅަ'],
    'kinship':  ['ދަރިފުޅު', 'ބޭބެ', 'ކާފަ', 'މާމަ', 'ފިރިމީހާ',
                 'އަނބިމީހާ', 'ކޮއްކޮ', 'މަންމަ', 'ބައްޕަ', 'ދައްތަ'],
    'clothing': ['އިސްޓާކީނު', 'ބޫޓު', 'ފަޓުލޫނު', 'ބުރުގާ', 'ފައިވާން',
                 'ސޯޓު', 'ހެދުން', 'ފޮތި', 'ލިބާސް', 'ގަމީސް'],
    'nature':   ['ތޫފާން', 'ކޯރު', 'މޫދު', 'ވައި', 'ވާރޭ',
                 'ވިލާ', 'ހަނދު', 'އުޑު', 'އިރު', 'ގަސް'],
}

# 4. ANALOGY GENERATORS
def gen_pair_analogies(pairs, label):
    analogies = []
    for i, (a, b) in enumerate(pairs):
        for j, (c, d) in enumerate(pairs):
            if i != j and a != c:
                analogies.append((a, b, c, d))
    return analogies

def gen_case_analogies(rows, col_a, col_b, label):
    pairs = []
    for row in rows:
        src, tgt = row[col_a], row[col_b]
        if src and tgt and src != "—" and tgt != "—":
            pairs.append((src, tgt))
    return gen_pair_analogies(pairs, label)

def gen_verb_analogies(rows, col_a, col_b, label):
    pairs = []
    for row in rows:
        src, tgt = row[col_a], row[col_b]
        if src and tgt:
            pairs.append((src, tgt))
    return gen_pair_analogies(pairs, label)

def build_all_categories():
    cats = {}
    cats["noun-plural-nonhuman"] = gen_pair_analogies(plural_nonhuman, "noun-plural-nonhuman")
    cats["noun-plural-human"]    = gen_pair_analogies(plural_human, "noun-plural-human")
    cats["noun-plural-assoc"] = gen_pair_analogies(plural_associative, "noun-plural-assoc")
    cats["kinship-gender"]  = gen_pair_analogies(kinship_gender,  "kinship-gender")
    for case_name, col in [("genitive", 2), ("dative", 3),
                            ("locative", 4), ("ablative", 5), ("sociative", 6)]:
        cats[f"noun-case-{case_name}"] = gen_case_analogies(non_human_cases, 0, col, f"noun-case-{case_name}")
    for case_name, col in [("genitive", 2), ("dative", 3), ("sociative", 4)]:
        cats[f"human-case-{case_name}"] = gen_case_analogies(human_cases, 0, col, f"human-case-{case_name}")
    for cat, (rows, ca, cb) in {
        "verb-present":   (finite_indicative, 0, 1),
        "verb-past":  (finite_indicative, 0, 2),
        "verb-past-progressive": (finite_indicative, 0, 3),
        "verb-future": (finite_indicative, 0, 4),
    }.items():
        cats[cat] = gen_verb_analogies(rows, ca, cb, cat)
    cats["verb-imperative"] = gen_verb_analogies(imperatives, 0, 1, "verb-imperative")
    cats["verb-imperative-polite"] = gen_verb_analogies(imperatives, 0, 2, "verb-imperative-polite")
    cats["verb-perfect"]   = gen_verb_analogies(perfect,   0, 1, "verb-perfect")
    cats["verb-potential"] = gen_verb_analogies(potential, 0, 1, "verb-potential")
    cats["verb-converb"] = gen_verb_analogies(converb,   0, 1, "verb-converb")
    cats["verb-honorific"] = gen_verb_analogies(honorific, 0, 1, "verb-honorific")
    return cats

#
# 5. ANALOGY EVALUATION
def evaluate_analogies(wv, topn=1, skip_oov=True, verbose=False):
    """Run morphology analogy evaluation. Returns aggregate accuracy + per-category."""
    all_cats = build_all_categories()
    vocab = set(wv.key_to_index.keys())
    print("\n" + "=" * 70)
    print("[1/3] MORPHOLOGY ANALOGY EVALUATION")
    print("=" * 70)
    print(f"{'CATEGORY':<35} {'ANALOGIES':>9} {'SKIPPED':>8} {'CORRECT':>8} {'ACC':>7}")
    print("=" * 70)
    summary = []
    for cat_name, analogies in all_cats.items():
        n_correct = n_total = n_skipped = 0
        wrong_examples = []
        for a, b, c, d in analogies:
            if skip_oov and not all(w in vocab for w in (a, b, c, d)):
                n_skipped += 1
                continue
            try:
                result = wv.most_similar(positive=[b, c], negative=[a], topn=topn)
                predicted = result[0][0] if result else None
            except KeyError:
                predicted = None
            if predicted == d:
                n_correct += 1
            elif verbose and len(wrong_examples) < 5:
                wrong_examples.append((a, b, c, d, predicted))
            n_total += 1
        acc = n_correct / n_total if n_total else 0.0
        summary.append((cat_name, len(analogies), n_skipped, n_correct, n_total, acc))
        print(f"  {cat_name:<33} {len(analogies):>9} {n_skipped:>8} {n_correct:>8} {acc:>6.1%}")
        if verbose:
            for a, b, c, d, predicted in wrong_examples:
                print(f"      {a} : {b} :: {c} → got '{predicted}', want '{d}'")
    total_an  = sum(s[1] for s in summary)
    total_ok  = sum(s[3] for s in summary)
    total_cnt = sum(s[4] for s in summary)
    overall   = total_ok / total_cnt if total_cnt else 0
    print("=" * 70)
    print(f"  {'OVERALL':<33} {total_an:>9}          {total_ok:>8} {overall:>6.1%}")
    print("=" * 70)
    return {
        'overall_acc': overall,
        'total_correct': total_ok,
        'total_evaluated': total_cnt,
        'per_category': {s[0]: s[5] for s in summary},
    }

# 6. OUTLIER DETECTION EVALUATION
def evaluate_outliers(wv, sets=None):
    """For each list, model picks word with lowest mean similarity to others."""
    if sets is None:
        sets = dhivehi_outlier_sets
    print("\n" + "=" * 70)
    print("[2/3] OUTLIER DETECTION EVALUATION")
    print("=" * 70)
    correct = 0
    skipped = 0
    n_total = len(sets)
    details = []
    for i, (words, expected_outlier) in enumerate(sets, 1):
        if not all(w in wv.key_to_index for w in words):
            missing = [w for w in words if w not in wv.key_to_index]
            print(f"  Set {i:2d}: SKIPPED  (OOV: {missing})")
            skipped += 1
            details.append({'words': words, 'expected': expected_outlier,
                            'predicted': None, 'correct': False, 'oov': True})
            continue
        vecs = np.array([wv[w] for w in words], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs_n = vecs / norms
        sim_matrix = vecs_n @ vecs_n.T
        np.fill_diagonal(sim_matrix, 0.0)
        mean_sims = sim_matrix.sum(axis=1) / (len(words) - 1)
        predicted_idx = int(np.argmin(mean_sims))
        predicted_word = words[predicted_idx]
        is_correct = (predicted_word == expected_outlier)
        if is_correct:
            correct += 1
            mark = '✓'
        else:
            mark = '✗'
        print(f"  Set {i:2d} {mark}  expected: {expected_outlier:<15}  predicted: {predicted_word}")
        details.append({'words': words, 'expected': expected_outlier,
                        'predicted': predicted_word, 'correct': is_correct,
                        'oov': False})
    n_evaluated = n_total - skipped
    accuracy = correct / n_evaluated if n_evaluated else 0.0
    print("=" * 70)
    print(f"  Sets evaluated : {n_evaluated}/{n_total}")
    print(f"  Skipped (OOV)  : {skipped}")
    print(f"  Accuracy       : {correct}/{n_evaluated} = {accuracy:.1%}")
    print("=" * 70)
    return {
        'accuracy': accuracy,
        'correct': correct,
        'evaluated': n_evaluated,
        'skipped': skipped,
        'details': details,
    }

# 7. WORD CATEGORIZATION (k-means PURITY)
def evaluate_categorization(wv, categories=None, random_state=42):
    """Cluster category words with k-means, measure purity vs ground truth."""
    if categories is None:
        categories = dhiv_categories
    print("\n" + "=" * 70)
    print("[3/3] WORD CATEGORIZATION PURITY")
    print("=" * 70)
    word_to_label = {}
    skipped_words = []
    for label, words in categories.items():
        for w in words:
            if w in wv.key_to_index:
                word_to_label[w] = label
            else:
                skipped_words.append((label, w))
    n_clusters = len(categories)
    n_words = len(word_to_label)
    if skipped_words:
        print(f"  OOV words (excluded from clustering):")
        for label, w in skipped_words:
            print(f"    {label}: {w}")
        print()
    if n_words < n_clusters * 2:
        print(f"  ✗ Too few in-vocab words ({n_words}) for {n_clusters}-way clustering. Skipping.")
        return {'purity': None, 'n_words': n_words, 'skipped': len(skipped_words)}
    words = list(word_to_label.keys())
    labels_true = [word_to_label[w] for w in words]
    vecs = np.array([wv[w] for w in words], dtype=np.float32)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = km.fit_predict(vecs)
    purity_correct = 0
    cluster_breakdown = defaultdict(Counter)
    for cluster_id, true_label in zip(cluster_ids, labels_true):
        cluster_breakdown[cluster_id][true_label] += 1
    for cluster_id, label_counts in cluster_breakdown.items():
        majority_count = max(label_counts.values())
        purity_correct += majority_count
    purity = purity_correct / n_words
    print(f"  Words evaluated  : {n_words}")
    print(f"  Categories (k)   : {n_clusters}")
    print(f"  K-means clusters :")
    for cluster_id in sorted(cluster_breakdown.keys()):
        counts = cluster_breakdown[cluster_id]
        total = sum(counts.values())
        majority_label, majority_n = counts.most_common(1)[0]
        breakdown = ', '.join(f"{lbl}={n}" for lbl, n in counts.most_common())
        print(f"    Cluster {cluster_id} (n={total}, majority={majority_label}/{majority_n}): {breakdown}")
    print("=" * 70)
    print(f"  Purity: {purity_correct}/{n_words} = {purity:.1%}")
    print(f"  (Random baseline ≈ {1/n_clusters:.1%})")
    print("=" * 70)
    return {
        'purity': purity,
        'n_words': n_words,
        'skipped': len(skipped_words),
        'cluster_breakdown': dict(cluster_breakdown),
    }

# 8. VOCAB DIAGNOSTICS
def diagnose_vocab(wv, n=50):
    vocab = list(wv.key_to_index.keys())
    print("\n" + "=" * 60)
    print(f"VOCAB SAMPLE (first {n} of {len(vocab):,} words)")
    print("=" * 60)
    for w in vocab[:n]:
        print(f"  {repr(w)}")
    print("=" * 60)

# 9. PIPELINE ENTRY POINT

def main(model_path=None, model_format=None):
    parser = argparse.ArgumentParser(description="Dhivehi intrinsic embedding evaluation.")
    parser.add_argument("--format", default=model_format,
                        choices=["binary", "text", "word2vec", "fasttext"])
    parser.add_argument("--topn", type=int, default=1)
    parser.add_argument("--no-skip-oov", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-analogy", action="store_true")
    parser.add_argument("--skip-outlier", action="store_true")
    parser.add_argument("--skip-categorization", action="store_true")
    args = parser.parse_args([])
    path, fmt = model_path, args.format
    print(f"\nLoading model: {path}  (format={fmt})")
    if fmt == "binary":
        from gensim.models import KeyedVectors
        wv = KeyedVectors.load_word2vec_format(path, binary=True)
    elif fmt == "text":
        from gensim.models import KeyedVectors
        wv = KeyedVectors.load_word2vec_format(path, binary=False)
    elif fmt == "word2vec":
        from gensim.models import Word2Vec
        wv = Word2Vec.load(path).wv
    elif fmt == "fasttext":
        from gensim.models import FastText
        wv = FastText.load(path).wv
    print(f"Vocabulary size: {len(wv.key_to_index):,}")
    print(f"Vector dim     : {wv.vector_size}")
    diagnose_vocab(wv)
    results = {}
    if not args.skip_analogy:
        results['analogy'] = evaluate_analogies(
            wv, topn=args.topn, skip_oov=not args.no_skip_oov, verbose=args.verbose)
    if not args.skip_outlier:
        results['outlier'] = evaluate_outliers(wv)
    if not args.skip_categorization:
        results['categorization'] = evaluate_categorization(wv)
    print("\n" + "=" * 70)
    print("INTRINSIC EVALUATION SUMMARY")
    print("=" * 70)
    if 'analogy' in results:
        a = results['analogy']
        print(f"  Morphology analogy : {a['overall_acc']:>6.1%}  "
              f"({a['total_correct']}/{a['total_evaluated']})")
    if 'outlier' in results:
        o = results['outlier']
        print(f"  Outlier detection  : {o['accuracy']:>6.1%}  "
              f"({o['correct']}/{o['evaluated']}, skipped {o['skipped']})")
    if 'categorization' in results:
        c = results['categorization']
        if c['purity'] is not None:
            print(f"  Categorization     : {c['purity']:>6.1%}  "
                  f"(purity, n={c['n_words']}, skipped {c['skipped']})")
        else:
            print(f"  Categorization     : SKIPPED (insufficient vocab)")
    print("=" * 70 + "\n")
    return results

def run_test_embed_dhiv(cfg: dict) -> None:
    main(model_path=cfg["dhiv_test_embedding"],
         model_format=cfg["embedding_type_dhiv"])

if __name__ == "__main__":
    raise SystemExit("must be run via the pipeline (d.test_embed step).")

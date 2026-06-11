#!/usr/bin/env python3
"""Extract plausible QUIP/GAP xml_label values from a GAP XML file.

Typical QUIP/LAMMPS usage is:
  pair_coeff * * model.xml "Potential xml_label=<LABEL>" 6

This helper prints candidate labels in priority order. It does not alter the XML.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def local_name(tag: str) -> str:
    return tag.rsplit('}', 1)[-1] if '}' in tag else tag


def candidate_labels(path: Path) -> list[str]:
    tree = ET.parse(path)
    root = tree.getroot()
    out: list[str] = []

    def add(value: str | None) -> None:
        if value and value not in out:
            out.append(value)

    # Highest-priority labels are usually on GAP_params / Potential-like elements.
    priority_tags = {'GAP_params', 'Potential', 'potential', 'GAP', 'gap'}
    for elem in tree.iter():
        name = local_name(elem.tag)
        if name in priority_tags:
            add(elem.attrib.get('label'))
            add(elem.attrib.get('xml_label'))
            add(elem.attrib.get('name'))

    # Root label can also be the correct one.
    add(root.attrib.get('label'))
    add(root.attrib.get('xml_label'))
    add(root.attrib.get('name'))

    # Fallback: all unique label attributes, preserving XML order.
    for elem in tree.iter():
        add(elem.attrib.get('label'))

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('xml', type=Path)
    parser.add_argument('--first', action='store_true', help='print only the first candidate')
    args = parser.parse_args()

    labels = candidate_labels(args.xml)
    if not labels:
        print(f'No label candidates found in {args.xml}', file=sys.stderr)
        return 1
    if args.first:
        print(labels[0])
    else:
        for label in labels:
            print(label)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

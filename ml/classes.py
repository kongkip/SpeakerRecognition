from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict
from generator import prepare_words_list


def get_classes(wanted_only=False):
    if wanted_only:
        classes = "benjamin_netanyau jens_stoltenberg julia_gillard magaret_tarcher nelson_mandela"
        classes = classes.split(' ')
        assert len(classes) == 5
    else:
        classes = ('benjamin_netanyau jens_stoltenberg julia_gillard magaret_tarcher nelson_mandela'
                   'other')
        classes = classes.split(' ')
        assert len(classes) == 30
    return classes


def get_int2label(wanted_only=False, extend_reversed=False):
    classes = get_classes(
        wanted_only=wanted_only)
    classes = prepare_words_list(classes)
    int2label = {i: l for i, l in enumerate(classes)}
    int2label = OrderedDict(sorted(int2label.items(), key=lambda x: x[0]))
    return int2label


def get_label2int(wanted_only=False, extend_reversed=False):
    classes = get_classes(
        wanted_only=wanted_only)
    classes = prepare_words_list(classes)
    label2int = {l: i for i, l in enumerate(classes)}
    label2int = OrderedDict(sorted(label2int.items(), key=lambda x: x[1]))
    return label2int

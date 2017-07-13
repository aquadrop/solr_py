#!/usr/bin/env python
# -*- coding: utf-8 -*-
import _uniout

chs_arabic_map = {u'零': 0, u'一': 1, u'二': 2, u'三': 3, u'四': 4,
                  u'五': 5, u'六': 6, u'七': 7, u'八': 8, u'九': 9,
                  u'十': 10, u'百': 100, u'千': 1000, u'万': 10000,
                  u'〇': 0, u'壹': 1, u'贰': 2, u'叁': 3, u'肆': 4,
                  u'伍': 5, u'陆': 6, u'柒': 7, u'捌': 8, u'玖': 9,
                  u'拾': 10, u'佰': 100, u'仟': 10000, u'萬': 10000,
                  u'亿': 100000000, u'億': 100000000, u'幺': 1,
                  u'０': 0, u'１': 1, u'２': 2, u'３': 3, u'４': 4,
                  u'５': 5, u'６': 6, u'７': 7, u'８': 8, u'９': 9, u'两': 2}

digit_list = [u'零', u'一', u'二', u'三', u'四',
              u'五', u'六', u'七', u'八', u'九',
              u'十', u'百', u'千', u'万',
              u'〇', u'壹', u'贰', u'叁', u'肆',
              u'伍', u'陆', u'柒', u'捌', u'玖',
              u'拾', u'佰', u'仟', u'萬',
              u'亿', u'億', u'幺', u'两']

lead_digits = [u'一', u'二', u'三', u'四',
              u'五', u'六', u'七', u'八', u'九',
              u'壹', u'贰', u'叁', u'肆',
              u'伍', u'陆', u'柒', u'捌', u'玖',
             u'两']


def cn2arab(chinese_digits):
    if len(chinese_digits) == 0:
        return False, ''

    chinese_digits = chinese_digits.decode("utf-8")

    prefix = []
    digit = []
    suffix = []
    pre_flag = False
    dig_flag = False
    for char in chinese_digits:
        if char not in digit_list and not pre_flag:
            prefix.append(char)
        elif char in digit_list and not dig_flag:
            digit.append(char)
            pre_flag = True
        else:
            dig_flag = True
            suffix.append(char)

    if len(digit) == 0:
        return False, ''.join(prefix)

    # print 'prefix', _uniout.unescape(str(prefix), 'utf-8')
    # print 'digit', _uniout.unescape(str(digit), 'utf-8')
    # print 'suffix', _uniout.unescape(str(suffix), 'utf-8')

    suffix = ''.join(suffix).encode('utf-8')

    transferred, suffix = cn2arab(suffix)
    return transferred or pre_flag, ''.join(prefix) + str(cn2arab_core(''.join(digit))) + suffix


def cn2arab_core(chinese_digits, encoding="utf-8"):

    if isinstance(chinese_digits, str):
        chinese_digits = chinese_digits.decode(encoding)

    if chinese_digits.isdigit():
        return float(chinese_digits)

    dig_mul = 1
    ## 100百万,取出100这个数字
    head_digits = []
    head = False
    for c in chinese_digits:
        if c.isdigit():
            head = True
            head_digits.append(c)
        else:
            break

    if len(head_digits) > 0:
        head_d = ''.join(head_digits)
        chinese_digits = chinese_digits.replace(head_d, '')
        dig_mul = float(head_d)

    if chinese_digits[0] not in lead_digits:
        chinese_digits = u'一' + chinese_digits
    result = 0
    tmp = 0
    hnd_mln = 0
    for count in range(len(chinese_digits)):
        curr_char = chinese_digits[count]
        curr_digit = chs_arabic_map.get(curr_char, None)
        # meet 「亿」 or 「億」
        if curr_digit == 10 ** 8:
            result = result + tmp
            result = result * curr_digit
            # get result before 「亿」 and store it into hnd_mln
            # reset `result`
            hnd_mln = hnd_mln * 10 ** 8 + result
            result = 0
            tmp = 0
        # meet 「万」 or 「萬」
        elif curr_digit == 10 ** 4:
            result = result + tmp
            result = result * curr_digit
            tmp = 0
        # meet 「十」, 「百」, 「千」 or their traditional version
        elif curr_digit >= 10:
            tmp = 1 if tmp == 0 else tmp
            result = result + curr_digit * tmp
            tmp = 0
        # meet single digit
        elif curr_digit is not None:
            tmp = tmp * 10 + curr_digit
        else:
            return result
    result = result + tmp
    result = result + hnd_mln
    return int(result * dig_mul)

if __name__ == '__main__':
    s = ['五十','三百','3百','两万','2万','2十万','100万','35','两千','39']
    for ss in s:
        print(ss, cn2arab_core(ss))
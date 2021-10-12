#!/usr/bin/python
import pymysql


class ColorInfo:
    def __init__(self, cursor):
        self.cur = cursor

    def get_colors(self):
        self.cur.execute("""SELECT color_id, color_name, color_type, color_code  FROM LegoSorterDB.Colors where parts_count > 99""")
        sqlresult = self.cur.fetchall()

        return_colors = []
        for row in sqlresult:
            return_colors.append(Color(row[0], row[1], row[2], row[3]))

        return return_colors


class Color:
    def __init__(self, color_id, color_name, color_type, color_code):
        self.color_id = color_id
        self.color_name = color_name
        self.color_type = color_type
        self.color_code = color_code

    def __str__(self):
        return "{} {}({}) [{}]".format(self.color_type, self.color_name, self.color_id, self.color_code)

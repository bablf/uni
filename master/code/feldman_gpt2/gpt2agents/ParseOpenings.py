import re
from typing import Pattern

import gpt2agents.utils.MySqlInterface as MSI

class Opening:
    site:str
    moves:str
    white:str
    black:str
    quote_regex:Pattern

    def __init__(self):
        self.reset()

    def reset(self):
        self.site = "unset"
        self.white = "unset"
        self.black = "unset"
        self.moves = "unset"
        self.quote_regex = re.compile('".*?"')

    def add_element(self, line:str) -> bool:
        if "Site" in line:
            s = self.quote_regex.findall(line)
            self.site = s[0].replace('"', '')
        elif "White" in line:
            s = self.quote_regex.findall(line)
            self.white = s[0].replace('"', '')
        elif "Black" in line:
            s = self.quote_regex.findall(line)
            self.black = s[0].replace('"', '')
        elif "1." in line:
            self.moves = line.lstrip("1.0 ")
            return True
        return False

    def to_sql(self):
        s = 'insert into table_openings (`site`, `white`, `black`, `moves`) values ("{}", "{}", "{}", "{}");'.format(
            self.site, self.white, self.black, self.moves)
        return s

class ParseOpenings:
    filename: str
    msi:MSI.MySqlInterface


    def __init__(self, filename:str):
        self.reset()
        self.filename = filename
        self.parse_file()

    def reset(self):
        self.filename = "unset"
        self.msi = MSI.MySqlInterface("root", "postgres", "gpt2_chess")


    def parse_file(self):
        print("parsing {}".format(self.filename))
        self.msi.write_data_get_row_id("truncate table_openings;")
        with open(self.filename) as f:
            count = 0
            po = Opening()
            for line in f:
                line = line.strip()
                # print("line {} = '{}'".format(count, line))
                if po.add_element(line):
                    s = po.to_sql()
                    count = self.msi.write_data_get_row_id(s)
                    print("{}: {}".format(count, s))
                    po = Opening()

        self.msi.close()

def main():
    po = ParseOpenings("../data/chess/eco.pgn")

if __name__ == "__main__":
    main()
#!/usr/bin/python
import pymysql


class DatabaseConnector:
    def __init__(self, configfile):
        try:
            self.connection = pymysql.connect(host=configfile['host'],  # your host, usually localhost
                                              user=configfile['user'],  # your username
                                              passwd=configfile['passwd'],  # your password
                                              db=configfile['db'],
                                              port=configfile['port'])
        except Exception as e:
            print(
                "Database not initialised? Run 'docker run --cap-add=sys_nice -d -p 3306:3306 mysql-server' in the "
                "'database'-folder of LegoImageCropper-Repo")
            raise

    def get_cursor(self):
        return self.connection.cursor()

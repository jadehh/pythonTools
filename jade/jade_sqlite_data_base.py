#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_sqlite_data_base.py
# @Author   : jade
# @Date     : 2021/11/27 15:57
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import sqlite3
import threading
import os
class JadeSqliteDataBase(object):
    def __init__(self,root_path,db_name,talbe_name, JadeLog=None):
        if os.path.exists(root_path) is False:
            os.mkdir(root_path)
        self._db_name = os.path.join(root_path, db_name)
        self.table_name = talbe_name
        self.JadeLog = JadeLog
        self.lock = threading.Lock()
        # 使用cursor()方法获取操作游标,连接数据库
        self.db = sqlite3.connect(self._db_name, check_same_thread=False)
        self.cursor = self.db.cursor()
        if self.JadeLog:
            self.JadeLog.DEBUG(
                "#" * 30 + "{}数据库连接成功".format(talbe_name) + "#" * 30
            )

        super(JadeSqliteDataBase, self).__init__()

        # 重新连接


    def create_table(self, table_config):
        """:parameter
        table_name:表名
        table_config:dict
        """
        sql_str = (
            "CREATE TABLE {} (id INTEGER PRIMARY KEY AUTOINCREMENT ,".format(
                self.table_name
            )
        )
        try:

            for key in table_config:
                if key == "rec_date":
                    continue
                if "path" in key:
                    size_str = " varchar(255),"
                elif key == "detect_type":
                    size_str = " varchar(255),"
                else:
                    size_str = " varchar(20),"
                sql_str = sql_str + key + size_str
            sql_str = (
                    sql_str
                    + "rec_date TIMESTAMP default (datetime('now', 'localtime')))"
            )
            self.cursor.execute(sql_str)
        except Exception as e:
            if "exists" in str(e):
                pass
            else:
                if self.JadeLog:
                    self.JadeLog.ERROR("创建表失败,失败原因为{}".format(e))


    def insert(self, data):
        """:插入一条数据
        data:插入的数据
        """
        self.db = sqlite3.connect(self._db_name, check_same_thread=False)
        self.cursor = self.db.cursor()
        sql_str = "INSERT OR IGNORE    INTO {} (".format(self.table_name)
        try:
            for data_key in data.keys():
                if type(data[data_key]) == str:
                    if len(data[data_key]) > 0:
                        sql_str = sql_str + data_key + ","
            sql_str = sql_str[:-1] + ") VALUES ("
            for data_key in data.keys():
                if type(data[data_key]) == str:
                    if len(data[data_key]) > 0:
                        sql_str = sql_str + "'{}'".format(data[data_key]) + ","
            sql_str = sql_str[:-1] + ")"
            self.lock.acquire()
            self.cursor.execute(sql_str)
            self.db.commit()
            self.lock.release()
        except Exception as e:
            if  self.JadeLog:
                self.JadeLog.ERROR("插入数据表失败,失败原因为{},sql语句为{}".format(e, sql_str))


    def query(self, start_time, end_time):
        """:查询所有的数据
        return:表单
        """
        self.db = sqlite3.connect(self._db_name, check_same_thread=False)
        self.cursor = self.db.cursor()
        sql_str = "SELECT * FROM {} where  rec_date >'{}' and rec_date<'{}'".format(self.table_name, start_time,
                                                                                    end_time)
        try:
            self.lock.acquire()
            self.cursor.execute(sql_str)
            results = self.cursor.fetchall()
            self.lock.release()
            return results
        except Exception as e:
            if self.JadeLog:
                self.JadeLog.ERROR("查询表失败,失败原因为{},sql语句为{}".format(e, sql_str))
            pass

    def clear(self):
        self.db = sqlite3.connect(self._db_name, check_same_thread=False)
        self.cursor = self.db.cursor()
        sql_str = "DELETE FROM {}".format(self.table_name)
        try:
            self.cursor.execute(sql_str)
            self.db.commit()
        except Exception as e:
            self.JadeLog.ERROR("插入数据表失败,失败原因为{},sql语句为{}".format(e, sql_str))




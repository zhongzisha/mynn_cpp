#include <string>

using namespace std;

#include <glog/logging.h>

#include "leveldb/db.h"
#include "leveldb/write_batch.h"
#include "lmdb.h"

#include "rocksdb/db.h"
#include "rocksdb/utilities/leveldb_options.h"
#include "rocksdb/write_batch.h"


// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

namespace db {

enum Mode { READ, WRITE, NEW };

class Cursor {
 public:
  Cursor() { }
  virtual ~Cursor() { }
  virtual void SeekToFirst() = 0;
  virtual void Next() = 0;
  virtual string key() = 0;
  virtual string value() = 0;
  virtual bool valid() = 0;

  DISABLE_COPY_AND_ASSIGN(Cursor);
};

class Transaction {
 public:
  Transaction() { }
  virtual ~Transaction() { }
  virtual void Put(const string& key, const string& value) = 0;
  virtual void Commit() = 0;

  DISABLE_COPY_AND_ASSIGN(Transaction);
};

class DB {
 public:
  DB() { }
  virtual ~DB() { }
  virtual void Open(const string& source, Mode mode) = 0;
  virtual void Close() = 0;
  virtual Cursor* NewCursor() = 0;
  virtual Transaction* NewTransaction() = 0;

  DISABLE_COPY_AND_ASSIGN(DB);
};

class LevelDBCursor : public Cursor {
 public:
  explicit LevelDBCursor(leveldb::Iterator* iter)
    : iter_(iter) { SeekToFirst(); }
  ~LevelDBCursor() { delete iter_; }
  virtual void SeekToFirst() { iter_->SeekToFirst(); }
  virtual void Next() { iter_->Next(); }
  virtual string key() { return iter_->key().ToString(); }
  virtual string value() { return iter_->value().ToString(); }
  virtual bool valid() { return iter_->Valid(); }

 private:
  leveldb::Iterator* iter_;
};

class LevelDBTransaction : public Transaction {
 public:
  explicit LevelDBTransaction(leveldb::DB* db) : db_(db) { CHECK_NOTNULL(db_); }
  virtual void Put(const string& key, const string& value) {
    batch_.Put(key, value);
  }
  virtual void Commit() {
    leveldb::Status status = db_->Write(leveldb::WriteOptions(), &batch_);
    CHECK(status.ok()) << "Failed to write batch to leveldb "
                       << std::endl << status.ToString();
  }

 private:
  leveldb::DB* db_;
  leveldb::WriteBatch batch_;

  DISABLE_COPY_AND_ASSIGN(LevelDBTransaction);
};

class LevelDB : public DB {
 public:
  LevelDB() : db_(NULL) { }
  virtual ~LevelDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (db_ != NULL) {
      delete db_;
      db_ = NULL;
    }
  }
  virtual LevelDBCursor* NewCursor() {
    return new LevelDBCursor(db_->NewIterator(leveldb::ReadOptions()));
  }
  virtual LevelDBTransaction* NewTransaction() {
    return new LevelDBTransaction(db_);
  }

 private:
  leveldb::DB* db_;
};

// for rocksdb
class RocksDBCursor : public Cursor {
 public:
  explicit RocksDBCursor(rocksdb::Iterator* iter)
    : iter_(iter) { SeekToFirst(); }
  ~RocksDBCursor() { delete iter_; }
  virtual void SeekToFirst() { iter_->SeekToFirst(); }
  virtual void Next() { iter_->Next(); }
  virtual string key() { return iter_->key().ToString(); }
  virtual string value() { return iter_->value().ToString(); }
  virtual bool valid() { return iter_->Valid(); }

 private:
  rocksdb::Iterator* iter_;
};

class RocksDBTransaction : public Transaction {
 public:
  explicit RocksDBTransaction(rocksdb::DB* db) : db_(db) { CHECK_NOTNULL(db_); }
  virtual void Put(const string& key, const string& value) {
    batch_.Put(key, value);
  }
  virtual void Commit() {
    rocksdb::Status status = db_->Write(rocksdb::WriteOptions(), &batch_);
    CHECK(status.ok()) << "Failed to write batch to rocksdb "
                       << std::endl << status.ToString();
  }

 private:
  rocksdb::DB* db_;
  rocksdb::WriteBatch batch_;

  DISABLE_COPY_AND_ASSIGN(RocksDBTransaction);
};

class RocksDB : public DB {
 public:
  RocksDB() : db_(NULL) { }
  virtual ~RocksDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (db_ != NULL) {
      delete db_;
      db_ = NULL;
    }
  }
  virtual RocksDBCursor* NewCursor() {
    return new RocksDBCursor(db_->NewIterator(rocksdb::ReadOptions()));
  }
  virtual RocksDBTransaction* NewTransaction() {
    return new RocksDBTransaction(db_);
  }

 private:
  rocksdb::DB* db_;
};


// for LMDB
inline void MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}

class LMDBCursor : public Cursor {
 public:
  explicit LMDBCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor)
    : mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false) {
    SeekToFirst();
  }
  virtual ~LMDBCursor() {
    mdb_cursor_close(mdb_cursor_);
    mdb_txn_abort(mdb_txn_);
  }
  virtual void SeekToFirst() { Seek(MDB_FIRST); }
  virtual void Next() { Seek(MDB_NEXT); }
  virtual string key() {
    return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  }
  virtual string value() {
    return string(static_cast<const char*>(mdb_value_.mv_data),
        mdb_value_.mv_size);
  }
  virtual bool valid() { return valid_; }

 private:
  void Seek(MDB_cursor_op op) {
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      valid_ = false;
    } else {
      MDB_CHECK(mdb_status);
      valid_ = true;
    }
  }

  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  bool valid_;
};

class LMDBTransaction : public Transaction {
 public:
  explicit LMDBTransaction(MDB_dbi* mdb_dbi, MDB_txn* mdb_txn)
    : mdb_dbi_(mdb_dbi), mdb_txn_(mdb_txn) { }
  virtual void Put(const string& key, const string& value);
  virtual void Commit() { MDB_CHECK(mdb_txn_commit(mdb_txn_)); }

 private:
  MDB_dbi* mdb_dbi_;
  MDB_txn* mdb_txn_;

  DISABLE_COPY_AND_ASSIGN(LMDBTransaction);
};

class LMDB : public DB {
 public:
  LMDB() : mdb_env_(NULL) { }
  virtual ~LMDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (mdb_env_ != NULL) {
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
      mdb_env_ = NULL;
    }
  }
  virtual LMDBCursor* NewCursor();
  virtual LMDBTransaction* NewTransaction();

 private:
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
};

DB* GetDB(const string& backend);

}  // namespace db


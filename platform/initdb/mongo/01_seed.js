// DataOps 카탈로그 ds_07_civil_complaints 의 MongoDB 대상 컬렉션.
// mongo 이미지의 /docker-entrypoint-initdb.d 는 최초 기동(빈 데이터 볼륨)에서 1회만 실행된다.
// MONGO_INITDB_DATABASE 로 지정된 DB 컨텍스트에서 돌아간다.

const collectionName = "col_civil_complaints";

if (!db.getCollectionNames().includes(collectionName)) {
  db.createCollection(collectionName);
}

// seq 는 range(25032~53024) 안에 들어와야 $gte/$lte 필터에 걸린다.
db[collectionName].createIndex({ seq: 1 }, { unique: true });
db[collectionName].createIndex({ region_code: 1, created_at: -1 });

if (db[collectionName].countDocuments({}) === 0) {
  const categories = ["교통", "주거", "보건", "복지", "환경"];
  const docs = [];
  for (let i = 0; i < 60; i += 1) {
    docs.push({
      seq: 25032 + i * 7,
      region_code: i % 2 === 0 ? "45190" : "46900",
      category: categories[i % categories.length],
      keyword_tags: ["인구감소", categories[i % categories.length], i % 3 === 0 ? "청년" : "고령"],
      sentiment_score: Number((-1 + (i % 20) * 0.1).toFixed(2)),
      created_at: new Date(Date.now() - i * 86400000),
    });
  }
  db[collectionName].insertMany(docs);
}

-- ID 1 (2020년 데이터)
UPDATE asset_metadata
SET search_tags = '{"data_type": "CSV_Timeseries", "topic": "restaurant_data", "data_year": 2020}'::jsonb
WHERE id = 1;

-- ID 2 (2021년 데이터)
UPDATE asset_metadata
SET search_tags = '{"data_type": "CSV_Timeseries", "topic": "restaurant_data", "data_year": 2021}'::jsonb
WHERE id = 2;

-- ID 3 (2022년 데이터)
UPDATE asset_metadata
SET search_tags = '{"data_type": "CSV_Timeseries", "topic": "restaurant_data", "data_year": 2022}'::jsonb
WHERE id = 3;

-- ID 4 (2023년 데이터)
UPDATE asset_metadata
SET search_tags = '{"data_type": "CSV_Timeseries", "topic": "restaurant_data", "data_year": 2023}'::jsonb
WHERE id = 4;

-- ID 5 (2024년 데이터)
UPDATE asset_metadata
SET search_tags = '{"data_type": "CSV_Timeseries", "topic": "restaurant_data", "data_year": 2024}'::jsonb
WHERE id = 5;

-- ID 6 (2025년 데이터)
UPDATE asset_metadata
SET search_tags = '{"data_type": "CSV_Timeseries", "topic": "restaurant_data", "data_year": 2025}'::jsonb
WHERE id = 6;

-- ID 7
UPDATE asset_metadata
SET search_tags = '
{
  "document_type": "PDF_Report",
  "report_type": "자금결산",
  "company": "케이비대덕위탁관리부동산투자회사",
  "fund_period": "1기",
  "report_date": "2025-11-28"
}'::jsonb
WHERE id = 7;

-- ID 8
UPDATE asset_metadata
SET search_tags = '
{
  "document_type": "PDF_Report",
  "report_type": "정기결산",
  "company": "대원제36호오피스위탁관리부동산투자회사",
  "fund_period": "1분기",
  "report_date": "2025-09-25"
}'::jsonb
WHERE id = 8;

-- ID 9
UPDATE asset_metadata
SET search_tags = '
{
  "document_type": "PDF_Report",
  "report_type": "정기결산",
  "company": "벨트라스트제18호위탁관리부동산투자회사",
  "fund_period": "4기_1분기",
  "report_date": "2025-09-30"
}'::jsonb
WHERE id = 9;

-- ID 10
UPDATE asset_metadata
SET search_tags = '
{
  "document_type": "PDF_Report",
  "report_type": "정기결산",
  "company": "코람코원광남제1호위탁관리부동산투자회사",
  "fund_period": "8기_1분기",
  "report_date": "2025-10-02"
}'::jsonb
WHERE id = 10;

-- ID 11
UPDATE asset_metadata
SET search_tags = '
{
  "document_type": "PDF_Report",
  "report_type": "정기결산",
  "company": "코람코라이프인프라위탁관리부동산투자회사",
  "fund_period": "11기_1분기",
  "report_date": "2025-10-15"
}'::jsonb
WHERE id = 11;
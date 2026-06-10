/**
 * 데이터 무결성 검사 서비스
 * DataOps 워크플로우에 필요한 데이터 검증 로직을 제공합니다.
 */

/**
 * 파일에 대한 데이터 무결성 검사
 * @param {Array<string>} filePaths - 검사할 파일 경로 배열
 * @returns {Promise<Object>} 검증 결과
 */
export const validateDataIntegrity = async (filePaths) => {
  // 실제로는 API를 통해 검증하지만, 여기서는 Mock 데이터 반환
  return new Promise((resolve) => {
    setTimeout(() => {
      // CSV 파일인 경우에만 검증 수행
      const csvFiles = filePaths.filter(path => path.endsWith('.csv'))
      
      if (csvFiles.length === 0) {
        resolve({
          status: 'skipped',
          message: '검증할 CSV 파일이 없습니다.',
          checks: []
        })
        return
      }

      // 각 파일에 대한 검증 결과 생성
      const checks = csvFiles.map(filePath => {
        // Mock 검증 로직
        const hasMissingValues = Math.random() > 0.7 // 30% 확률로 결측치 있음
        const rowCount = Math.floor(Math.random() * 10000) + 1000
        const previousRowCount = Math.floor(Math.random() * 10000) + 1000
        const rowCountChange = rowCount - previousRowCount
        const schemaChanged = Math.random() > 0.8 // 20% 확률로 스키마 변경

        return {
          filePath,
          missingValues: {
            status: hasMissingValues ? 'failed' : 'passed',
            count: hasMissingValues ? Math.floor(Math.random() * 100) : 0,
            message: hasMissingValues 
              ? `${Math.floor(Math.random() * 100)}개의 결측치가 발견되었습니다.`
              : '결측치가 없습니다.'
          },
          rowCount: {
            current: rowCount,
            previous: previousRowCount,
            change: rowCountChange,
            changePercent: ((rowCountChange / previousRowCount) * 100).toFixed(2)
          },
          schema: {
            changed: schemaChanged,
            message: schemaChanged 
              ? '스키마가 변경되었습니다.'
              : '스키마 변경 없음'
          }
        }
      })

      // 전체 검증 상태 결정
      const allPassed = checks.every(check => check.missingValues.status === 'passed')
      const hasSchemaChange = checks.some(check => check.schema.changed)

      resolve({
        status: allPassed ? 'passed' : 'failed',
        message: allPassed 
          ? '모든 검증을 통과했습니다.'
          : '일부 파일에서 문제가 발견되었습니다.',
        checks,
        summary: {
          totalFiles: csvFiles.length,
          passedFiles: checks.filter(c => c.missingValues.status === 'passed').length,
          failedFiles: checks.filter(c => c.missingValues.status === 'failed').length,
          totalRowCount: checks.reduce((sum, c) => sum + c.rowCount.current, 0),
          totalRowCountChange: checks.reduce((sum, c) => sum + c.rowCount.change, 0),
          hasSchemaChange
        }
      })
    }, 1500) // 검증 시뮬레이션 (1.5초)
  })
}

/**
 * 파일 메타데이터 조회
 * @param {Array<string>} filePaths - 파일 경로 배열
 * @returns {Promise<Object>} 메타데이터
 */
export const getFileMetadata = async (filePaths) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const csvFiles = filePaths.filter(path => path.endsWith('.csv'))
      
      const metadata = csvFiles.map(filePath => {
        const rowCount = Math.floor(Math.random() * 10000) + 1000
        const previousRowCount = Math.floor(Math.random() * 10000) + 1000
        const rowCountChange = rowCount - previousRowCount
        const schemaChanged = Math.random() > 0.8

        return {
          filePath,
          rowCount: {
            current: rowCount,
            previous: previousRowCount,
            change: rowCountChange,
            changePercent: ((rowCountChange / previousRowCount) * 100).toFixed(2)
          },
          schema: {
            changed: schemaChanged,
            columns: schemaChanged 
              ? ['column1', 'column2', 'column3', 'new_column']
              : ['column1', 'column2', 'column3']
          }
        }
      })

      resolve({
        files: metadata,
        summary: {
          totalRowCount: metadata.reduce((sum, m) => sum + m.rowCount.current, 0),
          totalRowCountChange: metadata.reduce((sum, m) => sum + m.rowCount.change, 0),
          hasSchemaChange: metadata.some(m => m.schema.changed),
          changedFiles: metadata.filter(m => m.schema.changed).length
        }
      })
    }, 500)
  })
}


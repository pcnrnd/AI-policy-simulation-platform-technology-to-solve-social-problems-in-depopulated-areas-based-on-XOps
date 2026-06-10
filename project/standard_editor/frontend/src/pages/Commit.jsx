import React, { useState, useEffect } from 'react'
import { useRepository } from '../contexts/RepositoryContext'
import { repositoryService } from '../services/repositoryService'
import { minioService } from '../services/minioService'
import { getFileTree } from '../services/fileTreeService'
import { validateDataIntegrity, getFileMetadata } from '../services/dataValidationService'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import FileTree from '../components/common/FileTree'
import { GitCommit, CheckCircle, XCircle, Package, File as FileIcon, AlertCircle, Database, TrendingUp, TrendingDown, Upload } from 'lucide-react'

const Commit = () => {
  const { repositoryPath } = useRepository()
  const [message, setMessage] = useState('')
  const [selectedFiles, setSelectedFiles] = useState([])
  const [fileTree, setFileTree] = useState([])
  const [push, setPush] = useState(false)
  const [loading, setLoading] = useState(false)
  const [loadingTree, setLoadingTree] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  // 데이터 무결성 검사 상태
  const [validationStatus, setValidationStatus] = useState('idle') // idle, checking, passed, failed
  const [validationResult, setValidationResult] = useState(null)
  const [metadata, setMetadata] = useState(null)
  
  // MinIO 업로드 관련 상태
  const [uploadToMinIO, setUploadToMinIO] = useState(false)
  const [minioBuckets, setMinioBuckets] = useState([])
  const [selectedMinioBucket, setSelectedMinioBucket] = useState('')

  // 파일 트리 로드
  useEffect(() => {
    if (repositoryPath) {
      loadFileTree()
    }
  }, [repositoryPath])
  
  // MinIO 버킷 목록 로드
  useEffect(() => {
    if (uploadToMinIO) {
      loadMinioBuckets()
    }
  }, [uploadToMinIO])
  
  const loadMinioBuckets = async () => {
    try {
      const response = await minioService.listBuckets()
      if (response.success && response.buckets) {
        setMinioBuckets(response.buckets)
        if (response.buckets.length > 0 && !selectedMinioBucket) {
          setSelectedMinioBucket(response.buckets[0].name)
        }
      }
    } catch (err) {
      console.error('MinIO 버킷 목록 조회 실패:', err)
    }
  }

  const loadFileTree = async () => {
    if (!repositoryPath) return
    
    setLoadingTree(true)
    try {
      const tree = await getFileTree(repositoryPath)
      setFileTree(tree)
    } catch (err) {
      setError('파일 목록을 불러오는데 실패했습니다.')
    } finally {
      setLoadingTree(false)
    }
  }

  const handleToggleFile = (filePath) => {
    setSelectedFiles((prev) => {
      const newFiles = prev.includes(filePath)
        ? prev.filter((path) => path !== filePath)
        : [...prev, filePath]
      
      // 파일 선택 시 검증 시작
      if (newFiles.length > 0 && newFiles.length !== prev.length) {
        runValidation(newFiles)
      } else if (newFiles.length === 0) {
        // 모든 파일 해제 시 검증 상태 초기화
        setValidationStatus('idle')
        setValidationResult(null)
        setMetadata(null)
      }
      
      return newFiles
    })
  }

  // 데이터 무결성 검사 실행
  const runValidation = async (files) => {
    setValidationStatus('checking')
    setValidationResult(null)
    setMetadata(null)

    try {
      // 검증 및 메타데이터를 병렬로 가져오기
      const [validation, fileMetadata] = await Promise.all([
        validateDataIntegrity(files),
        getFileMetadata(files)
      ])

      setValidationResult(validation)
      setMetadata(fileMetadata)
      setValidationStatus(validation.status === 'passed' ? 'passed' : 'failed')
    } catch (err) {
      console.error('검증 실패:', err)
      setValidationStatus('failed')
      setValidationResult({
        status: 'failed',
        message: '검증 중 오류가 발생했습니다.'
      })
    }
  }

  const handleCommit = async () => {
    if (!repositoryPath || !message) {
      setError('저장소 경로가 설정되지 않았거나 커밋 메시지를 입력해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // 1. 파일 추가 (DVC add)
      if (selectedFiles.length > 0) {
        await repositoryService.addFiles(selectedFiles, repositoryPath)
      }
      
      // 2. MinIO 업로드 (선택적)
      if (uploadToMinIO && selectedMinioBucket && selectedFiles.length > 0) {
        for (const filePath of selectedFiles) {
          try {
            // 로컬 파일 경로를 사용하여 업로드
            await minioService.uploadFileFromPath(
              selectedMinioBucket,
              filePath.replace(repositoryPath + '/', ''), // 상대 경로로 변환
              filePath
            )
          } catch (uploadErr) {
            console.error(`파일 업로드 실패 (${filePath}):`, uploadErr)
            // 개별 파일 업로드 실패는 경고만 표시하고 계속 진행
          }
        }
      }
      
      // 3. 커밋
      const response = await repositoryService.commit(message, repositoryPath, push)
      setResult(response)
      setError(null)
      setMessage('')
      setSelectedFiles([])
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '커밋에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleAddFiles = async () => {
    if (!repositoryPath) {
      setError('저장소 경로가 설정되지 않았습니다.')
      return
    }

    if (selectedFiles.length === 0) {
      setError('추가할 파일을 선택해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.addFiles(selectedFiles, repositoryPath)
      setResult(response)
      setError(null)
      setSelectedFiles([]) // 추가 후 선택 해제
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '파일 추가에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2 className="mb-3">Commit 관리</h2>
      
      {repositoryPath && (
        <div className="mb-3" style={{ 
          padding: '0.75rem', 
          background: 'var(--bg-secondary)', 
          borderRadius: '0.5rem',
          border: '1px solid var(--border-color)'
        }}>
          <strong>저장소 경로:</strong> <code>{repositoryPath}</code>
        </div>
      )}

      {!repositoryPath && (
        <div className="alert alert-error mb-3">
          저장소 경로가 설정되지 않았습니다. Home 페이지에서 경로를 설정해주세요.
        </div>
      )}

      <Card title="파일 추가">
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
          gap: '1rem' 
        }}>
          {/* 파일 트리 영역 */}
          <div>
            <div className="form-group">
              <label className="form-label">파일 목록</label>
              {loadingTree ? (
                <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                  파일 목록을 불러오는 중...
                </div>
              ) : (
                <FileTree
                  tree={fileTree}
                  selectedFiles={selectedFiles}
                  onToggleFile={handleToggleFile}
                />
              )}
            </div>
            <Button variant="secondary" onClick={loadFileTree} disabled={loadingTree} style={{ marginTop: '0.5rem' }}>
              새로고침
            </Button>
          </div>

          {/* Staging Area 영역 */}
          <div>
            <div className="form-group">
              <label className="form-label">
                <Package size={16} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
                Staging Area ({selectedFiles.length}개 파일)
              </label>
              <div
                style={{
                  border: '1px solid var(--border-color)',
                  borderRadius: '0.5rem',
                  padding: '0.75rem',
                  maxHeight: '400px',
                  overflowY: 'auto',
                  background: 'var(--bg-secondary)',
                  minHeight: '200px',
                }}
              >
                {selectedFiles.length === 0 ? (
                  <div style={{ 
                    padding: '2rem', 
                    textAlign: 'center', 
                    color: 'var(--text-secondary)',
                    fontStyle: 'italic'
                  }}>
                    선택된 파일이 없습니다.
                    <br />
                    왼쪽 트리에서 파일을 선택해주세요.
                  </div>
                ) : (
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                    {selectedFiles.map((filePath) => (
                      <li
                        key={filePath}
                        style={{
                          padding: '0.5rem',
                          marginBottom: '0.25rem',
                          background: 'var(--bg-tertiary)',
                          borderRadius: '0.25rem',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                        }}
                      >
                        <span style={{ fontSize: '0.875rem' }}>
                          <FileIcon size={14} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
                          {filePath}
                        </span>
                        <button
                          onClick={() => handleToggleFile(filePath)}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: 'var(--text-secondary)',
                            cursor: 'pointer',
                            padding: '0.25rem 0.5rem',
                            borderRadius: '0.25rem',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.background = 'var(--bg-primary)'
                            e.currentTarget.style.color = 'var(--text-primary)'
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.background = 'transparent'
                            e.currentTarget.style.color = 'var(--text-secondary)'
                          }}
                        >
                          ✕
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
              <Button variant="primary" onClick={handleAddFiles} disabled={loading || selectedFiles.length === 0}>
                선택한 파일 추가
              </Button>
              {selectedFiles.length > 0 && (
                <Button
                  variant="secondary"
                  onClick={() => setSelectedFiles([])}
                  disabled={loading}
                >
                  전체 해제
                </Button>
              )}
            </div>
          </div>
        </div>
      </Card>

      <Card title="변경사항 커밋">
        {/* 데이터 무결성 검사 패널 */}
        {selectedFiles.length > 0 && (
          <div className="validation-panel mb-3">
            <div className="form-group">
              <label className="form-label">
                <Database size={16} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
                데이터 무결성 검사
              </label>
              <div
                style={{
                  padding: '1rem',
                  background: 'var(--bg-tertiary)',
                  borderRadius: '0.5rem',
                  border: '1px solid var(--border-color)',
                  minHeight: '80px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {validationStatus === 'idle' && (
                  <div style={{ color: 'var(--text-secondary)', textAlign: 'center' }}>
                    파일을 선택하면 자동으로 검증이 시작됩니다.
                  </div>
                )}
                {validationStatus === 'checking' && (
                  <div style={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center',
                    gap: '0.5rem',
                    color: 'var(--text-secondary)'
                  }}>
                    <div className="spinner" style={{
                      width: '24px',
                      height: '24px',
                      border: '3px solid var(--border-color)',
                      borderTop: '3px solid var(--accent-primary)',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite'
                    }} />
                    <span>결측치 검사 중...</span>
                  </div>
                )}
                {validationStatus === 'passed' && validationResult && (
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '0.75rem',
                    color: 'var(--success)',
                    width: '100%',
                    justifyContent: 'center'
                  }}>
                    <CheckCircle size={20} />
                    <span style={{ fontWeight: 600 }}>✅ 통과</span>
                    <span style={{ 
                      marginLeft: '1rem', 
                      fontSize: '0.875rem',
                      color: 'var(--text-secondary)'
                    }}>
                      {validationResult.message}
                    </span>
                  </div>
                )}
                {validationStatus === 'failed' && validationResult && (
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '0.75rem',
                    color: 'var(--error)',
                    width: '100%',
                    justifyContent: 'center'
                  }}>
                    <AlertCircle size={20} />
                    <span style={{ fontWeight: 600 }}>검증 실패</span>
                    <span style={{ 
                      marginLeft: '1rem', 
                      fontSize: '0.875rem',
                      color: 'var(--text-secondary)'
                    }}>
                      {validationResult.message}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* 메타데이터 요약 카드 */}
        {metadata && metadata.summary && selectedFiles.length > 0 && (
          <div className="metadata-summary mb-3" style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '1rem'
          }}>
            <Card className="metadata-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <Database size={18} />
                <strong>Row Count 변화</strong>
              </div>
              <div style={{ 
                fontSize: '1.5rem', 
                fontWeight: 600,
                color: metadata.summary.totalRowCountChange >= 0 ? 'var(--success)' : 'var(--error)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                {metadata.summary.totalRowCountChange >= 0 ? (
                  <TrendingUp size={20} />
                ) : (
                  <TrendingDown size={20} />
                )}
                {metadata.summary.totalRowCountChange >= 0 ? '+' : ''}
                {metadata.summary.totalRowCountChange.toLocaleString()}
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                총 {metadata.summary.totalRowCount.toLocaleString()}개 행
              </div>
            </Card>

            <Card className="metadata-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <FileIcon size={18} />
                <strong>Schema 변경 여부</strong>
              </div>
              <div style={{ 
                fontSize: '1.25rem', 
                fontWeight: 600,
                color: metadata.summary.hasSchemaChange ? 'var(--warning)' : 'var(--success)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                {metadata.summary.hasSchemaChange ? (
                  <>
                    <AlertCircle size={20} />
                    <span>변경됨</span>
                  </>
                ) : (
                  <>
                    <CheckCircle size={20} />
                    <span>변경 없음</span>
                  </>
                )}
              </div>
              {metadata.summary.hasSchemaChange && (
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  {metadata.summary.changedFiles}개 파일의 스키마가 변경되었습니다.
                </div>
              )}
            </Card>
          </div>
        )}

        <div className="form-group">
          <label className="form-label">커밋 메시지 *</label>
          <textarea
            className="form-input form-textarea"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="커밋 메시지를 입력하세요..."
          />
        </div>
        <div className="form-group">
          <label className="form-label">
            <input
              type="checkbox"
              className="form-checkbox"
              checked={push}
              onChange={(e) => setPush(e.target.checked)}
            />
            원격 저장소로 푸시
          </label>
        </div>
        
        <div className="form-group">
          <label className="form-label">
            <input
              type="checkbox"
              className="form-checkbox"
              checked={uploadToMinIO}
              onChange={(e) => setUploadToMinIO(e.target.checked)}
            />
            <Upload size={16} style={{ marginLeft: '0.5rem', verticalAlign: 'middle' }} />
            MinIO에 직접 업로드
          </label>
        </div>
        
        {uploadToMinIO && (
          <div className="form-group">
            <label className="form-label">MinIO 버킷 선택</label>
            {minioBuckets.length > 0 ? (
              <select
                className="form-input"
                value={selectedMinioBucket}
                onChange={(e) => setSelectedMinioBucket(e.target.value)}
              >
                {minioBuckets.map((bucket) => (
                  <option key={bucket.name} value={bucket.name}>
                    {bucket.name}
                  </option>
                ))}
              </select>
            ) : (
              <div className="alert alert-warning">
                버킷을 불러올 수 없습니다. MinIO 관리 페이지에서 버킷을 생성해주세요.
              </div>
            )}
          </div>
        )}
        
        <Button 
          variant="primary" 
          onClick={handleCommit} 
          disabled={loading || (selectedFiles.length > 0 && validationStatus !== 'passed') || (uploadToMinIO && !selectedMinioBucket)}
        >
          <GitCommit size={16} />
          커밋
        </Button>
        {selectedFiles.length > 0 && validationStatus !== 'passed' && validationStatus !== 'idle' && (
          <div style={{ 
            marginTop: '0.5rem', 
            fontSize: '0.875rem', 
            color: 'var(--text-secondary)',
            fontStyle: 'italic'
          }}>
            * 데이터 무결성 검사를 통과해야 커밋할 수 있습니다.
          </div>
        )}
      </Card>

      {error && (
        <div className="alert alert-error">
          <XCircle size={16} />
          {error}
        </div>
      )}

      {result && result.success && (
        <div className="alert alert-success">
          <CheckCircle size={16} />
          작업이 성공적으로 완료되었습니다.
        </div>
      )}

      {result && (
        <Card title="결과">
          <pre style={{ 
            background: 'var(--bg-tertiary)', 
            padding: '1rem', 
            borderRadius: '0.5rem',
            overflow: 'auto',
            fontSize: '0.875rem'
          }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </Card>
      )}

      {loading && <div className="loading">처리 중...</div>}
    </div>
  )
}

export default Commit


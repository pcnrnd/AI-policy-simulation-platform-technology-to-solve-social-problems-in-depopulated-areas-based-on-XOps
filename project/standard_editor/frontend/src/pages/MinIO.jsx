import React, { useState, useEffect } from 'react'
import { minioService } from '../services/minioService'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import { Database, Upload, Download, Trash2, Folder, File as FileIcon, CheckCircle, XCircle, RefreshCw } from 'lucide-react'

const MinIO = () => {
  const [buckets, setBuckets] = useState([])
  const [selectedBucket, setSelectedBucket] = useState('')
  const [objects, setObjects] = useState([])
  const [newBucketName, setNewBucketName] = useState('')
  const [uploadFile, setUploadFile] = useState(null)
  const [uploadObjectName, setUploadObjectName] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  // 버킷 목록 로드
  useEffect(() => {
    loadBuckets()
  }, [])

  // 선택된 버킷의 객체 목록 로드
  useEffect(() => {
    if (selectedBucket) {
      loadObjects(selectedBucket)
    } else {
      setObjects([])
    }
  }, [selectedBucket])

  const loadBuckets = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await minioService.listBuckets()
      if (response.success && response.buckets) {
        setBuckets(response.buckets)
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '버킷 목록 조회에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const loadObjects = async (bucketName) => {
    setLoading(true)
    setError(null)
    try {
      const response = await minioService.listObjects(bucketName)
      if (response.success && response.objects) {
        setObjects(response.objects)
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '객체 목록 조회에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleCreateBucket = async () => {
    if (!newBucketName) {
      setError('버킷 이름을 입력해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await minioService.createBucket(newBucketName)
      setResult(response)
      setNewBucketName('')
      await loadBuckets()
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '버킷 생성에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleUploadFile = async () => {
    if (!selectedBucket || !uploadFile || !uploadObjectName) {
      setError('버킷, 파일, 객체 이름을 모두 입력해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await minioService.uploadFile(selectedBucket, uploadObjectName, uploadFile)
      setResult(response)
      setUploadFile(null)
      setUploadObjectName('')
      await loadObjects(selectedBucket)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '파일 업로드에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteObject = async (objectName) => {
    if (!selectedBucket || !objectName) {
      setError('버킷과 객체 이름이 필요합니다.')
      return
    }

    if (!window.confirm(`정말로 "${objectName}" 파일을 삭제하시겠습니까?`)) {
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await minioService.deleteFile(selectedBucket, objectName)
      setResult(response)
      await loadObjects(selectedBucket)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '파일 삭제에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div>
      <h2 className="mb-3">MinIO 관리</h2>

      <Card title="버킷 관리">
        <div className="form-group">
          <label className="form-label">새 버킷 이름</label>
          <div className="flex gap-2">
            <input
              type="text"
              className="form-input"
              value={newBucketName}
              onChange={(e) => setNewBucketName(e.target.value)}
              placeholder="bucket-name"
            />
            <Button variant="primary" onClick={handleCreateBucket} disabled={loading || !newBucketName}>
              <Database size={16} />
              버킷 생성
            </Button>
            <Button variant="secondary" onClick={loadBuckets} disabled={loading}>
              <RefreshCw size={16} />
              새로고침
            </Button>
          </div>
        </div>

        {buckets.length > 0 && (
          <div className="mt-3">
            <label className="form-label">버킷 선택</label>
            <select
              className="form-input"
              value={selectedBucket}
              onChange={(e) => setSelectedBucket(e.target.value)}
            >
              <option value="">버킷을 선택하세요</option>
              {buckets.map((bucket) => (
                <option key={bucket.name} value={bucket.name}>
                  {bucket.name}
                </option>
              ))}
            </select>
          </div>
        )}

        {buckets.length === 0 && !loading && (
          <div className="alert alert-warning mt-3">
            버킷이 없습니다. 새 버킷을 생성해주세요.
          </div>
        )}
      </Card>

      {selectedBucket && (
        <Card title={`${selectedBucket} 버킷 - 파일 관리`}>
          <div className="form-group">
            <label className="form-label">파일 업로드</label>
            <div className="form-group">
              <input
                type="file"
                className="form-input"
                onChange={(e) => {
                  setUploadFile(e.target.files[0])
                  if (e.target.files[0] && !uploadObjectName) {
                    setUploadObjectName(e.target.files[0].name)
                  }
                }}
              />
            </div>
            <div className="form-group">
              <label className="form-label">객체 이름 (파일 경로)</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  className="form-input"
                  value={uploadObjectName}
                  onChange={(e) => setUploadObjectName(e.target.value)}
                  placeholder="path/to/file.csv"
                />
                <Button variant="success" onClick={handleUploadFile} disabled={loading || !uploadFile || !uploadObjectName}>
                  <Upload size={16} />
                  업로드
                </Button>
              </div>
            </div>
          </div>

          <div className="mt-3">
            <div className="flex justify-between align-center mb-2">
              <strong>파일 목록 ({objects.length}개)</strong>
              <Button variant="secondary" size="small" onClick={() => loadObjects(selectedBucket)} disabled={loading}>
                <RefreshCw size={14} />
                새로고침
              </Button>
            </div>
            {objects.length > 0 ? (
              <div style={{ 
                border: '1px solid var(--border-color)', 
                borderRadius: '0.5rem',
                overflow: 'auto',
                maxHeight: '400px'
              }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ background: 'var(--bg-tertiary)', borderBottom: '1px solid var(--border-color)' }}>
                      <th style={{ padding: '0.75rem', textAlign: 'left' }}>파일명</th>
                      <th style={{ padding: '0.75rem', textAlign: 'right' }}>크기</th>
                      <th style={{ padding: '0.75rem', textAlign: 'left' }}>수정일</th>
                      <th style={{ padding: '0.75rem', textAlign: 'center' }}>작업</th>
                    </tr>
                  </thead>
                  <tbody>
                    {objects.map((obj) => (
                      <tr key={obj.name} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <td style={{ padding: '0.75rem' }}>
                          <FileIcon size={16} style={{ display: 'inline', marginRight: '0.5rem', verticalAlign: 'middle' }} />
                          {obj.name}
                        </td>
                        <td style={{ padding: '0.75rem', textAlign: 'right' }}>
                          {formatFileSize(obj.size)}
                        </td>
                        <td style={{ padding: '0.75rem' }}>
                          {obj.last_modified ? new Date(obj.last_modified).toLocaleString('ko-KR') : '-'}
                        </td>
                        <td style={{ padding: '0.75rem', textAlign: 'center' }}>
                          <Button
                            variant="danger"
                            size="small"
                            onClick={() => handleDeleteObject(obj.name)}
                            disabled={loading}
                          >
                            <Trash2 size={14} />
                            삭제
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="alert alert-info mt-3">
                파일이 없습니다.
              </div>
            )}
          </div>
        </Card>
      )}

      {error && (
        <div className="alert alert-error">
          <XCircle size={16} />
          {error}
        </div>
      )}

      {result && result.success && (
        <div className="alert alert-success">
          <CheckCircle size={16} />
          {result.message || '작업이 성공적으로 완료되었습니다.'}
        </div>
      )}

      {loading && <div className="loading">처리 중...</div>}
    </div>
  )
}

export default MinIO


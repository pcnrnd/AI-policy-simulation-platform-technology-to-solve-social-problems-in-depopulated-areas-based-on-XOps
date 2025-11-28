import React, { useState } from 'react'
import { repositoryService } from '../services/repositoryService'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import { GitBranch, CheckCircle, XCircle } from 'lucide-react'

const Branch = () => {
  const [path, setPath] = useState('')
  const [branchName, setBranchName] = useState('')
  const [sourceBranch, setSourceBranch] = useState('')
  const [targetBranch, setTargetBranch] = useState('')
  const [createNew, setCreateNew] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [branches, setBranches] = useState(null)

  const handleCheckout = async () => {
    if (!path || !branchName) {
      setError('경로와 브랜치 이름을 입력해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.checkout(branchName, path, createNew)
      setResult(response)
      setError(null)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '체크아웃에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleMerge = async () => {
    if (!path || !sourceBranch || !targetBranch) {
      setError('경로, 소스 브랜치, 타겟 브랜치를 모두 입력해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.merge(sourceBranch, targetBranch, path)
      setResult(response)
      setError(null)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '병합에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleGetBranches = async () => {
    if (!path) {
      setError('경로를 입력해주세요.')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await repositoryService.getBranches(path)
      setBranches(response)
      setError(null)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '브랜치 목록 조회에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2 className="mb-3">Branch 관리</h2>

      <Card title="브랜치 체크아웃">
        <div className="form-group">
          <label className="form-label">저장소 경로 *</label>
          <input
            type="text"
            className="form-input"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/path/to/repository"
          />
        </div>
        <div className="form-group">
          <label className="form-label">브랜치 이름 *</label>
          <input
            type="text"
            className="form-input"
            value={branchName}
            onChange={(e) => setBranchName(e.target.value)}
            placeholder="branch-name"
          />
        </div>
        <div className="form-group">
          <label className="form-label">
            <input
              type="checkbox"
              className="form-checkbox"
              checked={createNew}
              onChange={(e) => setCreateNew(e.target.checked)}
            />
            새 브랜치 생성
          </label>
        </div>
        <Button variant="primary" onClick={handleCheckout} disabled={loading}>
          <GitBranch size={16} />
          체크아웃
        </Button>
      </Card>

      <Card title="브랜치 병합">
        <div className="form-group">
          <label className="form-label">저장소 경로 *</label>
          <input
            type="text"
            className="form-input"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/path/to/repository"
          />
        </div>
        <div className="form-group">
          <label className="form-label">소스 브랜치 *</label>
          <input
            type="text"
            className="form-input"
            value={sourceBranch}
            onChange={(e) => setSourceBranch(e.target.value)}
            placeholder="source-branch"
          />
        </div>
        <div className="form-group">
          <label className="form-label">타겟 브랜치 *</label>
          <input
            type="text"
            className="form-input"
            value={targetBranch}
            onChange={(e) => setTargetBranch(e.target.value)}
            placeholder="target-branch"
          />
        </div>
        <Button variant="primary" onClick={handleMerge} disabled={loading}>
          병합
        </Button>
      </Card>

      <Card title="브랜치 목록">
        <div className="form-group">
          <label className="form-label">저장소 경로 *</label>
          <input
            type="text"
            className="form-input"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/path/to/repository"
          />
        </div>
        <Button variant="secondary" onClick={handleGetBranches} disabled={loading}>
          목록 조회
        </Button>

        {branches && (
          <div className="mt-3">
            <pre style={{ 
              background: 'var(--bg-tertiary)', 
              padding: '1rem', 
              borderRadius: '0.5rem',
              overflow: 'auto',
              fontSize: '0.875rem'
            }}>
              {JSON.stringify(branches, null, 2)}
            </pre>
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

export default Branch


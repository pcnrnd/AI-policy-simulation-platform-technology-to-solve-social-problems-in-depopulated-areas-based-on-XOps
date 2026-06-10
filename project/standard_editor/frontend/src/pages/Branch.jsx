import React, { useState, useEffect } from 'react'
import { useRepository } from '../contexts/RepositoryContext'
import { repositoryService } from '../services/repositoryService'
import Card from '../components/common/Card'
import Button from '../components/common/Button'
import Select from '../components/common/Select'
import { GitBranch, CheckCircle, XCircle, ArrowRight, GitMerge } from 'lucide-react'

const Branch = () => {
  const { repositoryPath } = useRepository()
  const [branchName, setBranchName] = useState('')
  const [sourceBranch, setSourceBranch] = useState('')
  const [targetBranch, setTargetBranch] = useState('')
  const [createNew, setCreateNew] = useState(false)
  const [loading, setLoading] = useState(false)
  const [loadingBranches, setLoadingBranches] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [branches, setBranches] = useState([])
  const [branchList, setBranchList] = useState([])

  // 브랜치 목록 자동 로드
  useEffect(() => {
    if (repositoryPath) {
      loadBranches()
    }
  }, [repositoryPath])

  // 브랜치 목록 파싱
  useEffect(() => {
    if (branches && branches.data) {
      parseBranchList(branches)
    }
  }, [branches])

  const parseBranchList = (branchData) => {
    try {
      // API 응답에서 브랜치 목록 추출
      let branchNames = []
      
      if (branchData.data && branchData.data.data) {
        // stdout에서 브랜치 목록 파싱
        const stdout = branchData.data.data.stdout || branchData.data.data
        if (typeof stdout === 'string') {
          branchNames = stdout
            .split('\n')
            .map(line => line.trim().replace(/^\*\s*/, '').replace(/^remotes\/[^\/]+\//, ''))
            .filter(line => line && !line.includes('HEAD'))
        } else if (Array.isArray(stdout)) {
          branchNames = stdout
        } else if (branchData.data.data.data) {
          // 다른 형식의 응답 처리
          branchNames = Array.isArray(branchData.data.data.data) 
            ? branchData.data.data.data 
            : []
        }
      }
      
      // 중복 제거 및 정렬
      branchNames = [...new Set(branchNames)].filter(b => b).sort()
      setBranchList(branchNames)
    } catch (err) {
      console.error('브랜치 목록 파싱 실패:', err)
      setBranchList([])
    }
  }

  const loadBranches = async () => {
    if (!repositoryPath) return

    setLoadingBranches(true)
    setError(null)

    try {
      const response = await repositoryService.getBranches(repositoryPath)
      setBranches(response)
    } catch (err) {
      console.error('브랜치 목록 로드 실패:', err)
      // 에러가 발생해도 계속 진행 (빈 목록 사용)
      setBranchList([])
    } finally {
      setLoadingBranches(false)
    }
  }

  const handleCheckout = async () => {
    if (!repositoryPath || !branchName) {
      setError('저장소 경로가 설정되지 않았거나 브랜치를 선택해주세요.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.checkout(branchName, repositoryPath, createNew)
      setResult(response)
      setError(null)
      // 성공 시 브랜치 목록 새로고침
      if (createNew) {
        loadBranches()
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '체크아웃에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const handleMerge = async () => {
    if (!repositoryPath || !sourceBranch || !targetBranch) {
      setError('저장소 경로가 설정되지 않았거나 소스 브랜치, 타겟 브랜치를 모두 선택해주세요.')
      return
    }

    if (sourceBranch === targetBranch) {
      setError('소스 브랜치와 타겟 브랜치는 같을 수 없습니다.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await repositoryService.merge(sourceBranch, targetBranch, repositoryPath)
      setResult(response)
      setError(null)
      // 병합 후 브랜치 목록 새로고침
      loadBranches()
    } catch (err) {
      setError(err.response?.data?.detail || err.message || '병합에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2 className="mb-3">Branch 관리</h2>
      
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

      <Card title="브랜치 체크아웃">
        <div className="form-group">
          <label className="form-label">브랜치 선택 *</label>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <div style={{ flex: 1 }}>
              <Select
                value={branchName}
                onChange={(e) => setBranchName(e.target.value)}
                options={branchList}
                placeholder={loadingBranches ? '로딩 중...' : '브랜치를 선택하세요'}
                disabled={loadingBranches || !repositoryPath}
              />
            </div>
            <Button 
              variant="secondary" 
              onClick={loadBranches} 
              disabled={loadingBranches || !repositoryPath}
              style={{ whiteSpace: 'nowrap' }}
            >
              새로고침
            </Button>
          </div>
        </div>
        <div className="form-group">
          <label className="form-label">
            <input
              type="checkbox"
              className="form-checkbox"
              checked={createNew}
              onChange={(e) => {
                setCreateNew(e.target.checked)
                if (e.target.checked) {
                  setBranchName('')
                }
              }}
            />
            새 브랜치 생성 (체크 시 브랜치 이름을 직접 입력할 수 있습니다)
          </label>
        </div>
        {createNew && (
          <div className="form-group">
            <label className="form-label">새 브랜치 이름 *</label>
            <input
              type="text"
              className="form-input"
              value={branchName}
              onChange={(e) => setBranchName(e.target.value)}
              placeholder="new-branch-name"
            />
          </div>
        )}
        <Button variant="primary" onClick={handleCheckout} disabled={loading || !branchName}>
          <GitBranch size={16} />
          체크아웃
        </Button>
      </Card>

      {/* Merge UI - 별도 섹션으로 분리 */}
      <Card title="브랜치 병합" className="merge-section">
        <div className="merge-container">
          <div className="merge-source">
            <label className="form-label">
              <GitBranch size={16} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
              소스 브랜치 (Source)
            </label>
            <Select
              value={sourceBranch}
              onChange={(e) => setSourceBranch(e.target.value)}
              options={branchList.filter(b => b !== targetBranch)}
              placeholder={loadingBranches ? '로딩 중...' : '병합할 소스 브랜치 선택'}
              disabled={loadingBranches || !repositoryPath}
            />
            {sourceBranch && (
              <div className="merge-branch-display">
                <code>{sourceBranch}</code>
              </div>
            )}
          </div>

          <div className="merge-arrow">
            <ArrowRight size={32} />
          </div>

          <div className="merge-target">
            <label className="form-label">
              <GitMerge size={16} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
              타겟 브랜치 (Target)
            </label>
            <Select
              value={targetBranch}
              onChange={(e) => setTargetBranch(e.target.value)}
              options={branchList.filter(b => b !== sourceBranch)}
              placeholder={loadingBranches ? '로딩 중...' : '병합될 타겟 브랜치 선택'}
              disabled={loadingBranches || !repositoryPath}
            />
            {targetBranch && (
              <div className="merge-branch-display">
                <code>{targetBranch}</code>
              </div>
            )}
          </div>
        </div>

        <div className="merge-info" style={{ 
          marginTop: '1.5rem', 
          padding: '1rem', 
          background: 'var(--bg-tertiary)', 
          borderRadius: '0.5rem',
          border: '1px solid var(--border-color)'
        }}>
          <p style={{ margin: 0, fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            <strong>병합 정보:</strong> {sourceBranch && targetBranch 
              ? `"${sourceBranch}" 브랜치의 변경사항을 "${targetBranch}" 브랜치로 병합합니다.`
              : '소스 브랜치와 타겟 브랜치를 모두 선택해주세요.'}
          </p>
        </div>

        <div style={{ marginTop: '1.5rem' }}>
          <Button 
            variant="primary" 
            onClick={handleMerge} 
            disabled={loading || !sourceBranch || !targetBranch || sourceBranch === targetBranch}
          >
            <GitMerge size={16} />
            병합 실행
          </Button>
        </div>
      </Card>

      <Card title="브랜치 목록">
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
          <Button variant="secondary" onClick={loadBranches} disabled={loadingBranches || !repositoryPath}>
            목록 새로고침
          </Button>
        </div>

        {loadingBranches ? (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
            브랜치 목록을 불러오는 중...
          </div>
        ) : branchList.length > 0 ? (
          <div style={{ 
            background: 'var(--bg-tertiary)', 
            padding: '1rem', 
            borderRadius: '0.5rem',
            border: '1px solid var(--border-color)'
          }}>
            <ul style={{ 
              listStyle: 'none', 
              padding: 0, 
              margin: 0,
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem'
            }}>
              {branchList.map((branch) => (
                <li
                  key={branch}
                  style={{
                    padding: '0.75rem',
                    background: 'var(--bg-secondary)',
                    borderRadius: '0.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                  }}
                >
                  <GitBranch size={16} />
                  <code style={{ fontSize: '0.875rem' }}>{branch}</code>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <div style={{ 
            padding: '2rem', 
            textAlign: 'center', 
            color: 'var(--text-secondary)',
            fontStyle: 'italic'
          }}>
            브랜치가 없거나 목록을 불러올 수 없습니다.
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

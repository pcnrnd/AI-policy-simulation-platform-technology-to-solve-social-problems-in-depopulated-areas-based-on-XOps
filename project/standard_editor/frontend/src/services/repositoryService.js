import api from './api'

export const repositoryService = {
  /**
   * Repository 초기화
   */
  async initRepository(path, remoteUrl = null) {
    const response = await api.post('/api/v1/repository/init', {
      path,
      remote_url: remoteUrl,
    })
    return response.data
  },

  /**
   * 브랜치 체크아웃
   */
  async checkout(branch, path, create = false) {
    const response = await api.post('/api/v1/repository/checkout', {
      branch,
      path,
      create,
    })
    return response.data
  },

  /**
   * 변경사항 커밋
   */
  async commit(message, path, push = false) {
    const response = await api.post('/api/v1/repository/commit', {
      message,
      path,
      push,
    })
    return response.data
  },

  /**
   * 파일 추가
   */
  async addFiles(files, path) {
    const response = await api.post('/api/v1/repository/add', {
      files,
      path,
    })
    return response.data
  },

  /**
   * 원격 저장소에서 업데이트
   */
  async update(path) {
    const response = await api.post('/api/v1/repository/update', {
      path,
    })
    return response.data
  },

  /**
   * 브랜치 병합
   */
  async merge(sourceBranch, targetBranch, path) {
    const response = await api.post('/api/v1/repository/merge', {
      source_branch: sourceBranch,
      target_branch: targetBranch,
      path,
    })
    return response.data
  },

  /**
   * 브랜치 목록 조회
   */
  async getBranches(path) {
    const response = await api.get('/api/v1/repository/branches', {
      params: { path },
    })
    return response.data
  },

  /**
   * 저장소 상태 조회
   */
  async getStatus(path) {
    const response = await api.get('/api/v1/repository/status', {
      params: { path },
    })
    return response.data
  },
}


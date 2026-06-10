import api from './api'

/**
 * GitHub 관련 API 서비스
 */
export const githubService = {
  /**
   * GitHub 원격 저장소 설정
   * @param {string} path - 저장소 경로
   * @param {string} remoteName - 원격 저장소 이름 (기본값: origin)
   * @param {string} remoteUrl - GitHub 저장소 URL
   */
  async setRemote(path, remoteName = 'origin', remoteUrl) {
    const response = await api.post('/api/v1/github/remote', {
      path,
      remote_name: remoteName,
      remote_url: remoteUrl,
    })
    return response.data
  },

  /**
   * GitHub 원격 저장소 목록 조회
   * @param {string} path - 저장소 경로
   */
  async getRemotes(path) {
    const response = await api.get('/api/v1/github/remote', {
      params: { path },
    })
    return response.data
  },

  /**
   * GitHub에 푸시
   * @param {string} path - 저장소 경로
   * @param {string} remoteName - 원격 저장소 이름 (기본값: origin)
   * @param {string} branch - 푸시할 브랜치 (선택)
   */
  async push(path, remoteName = 'origin', branch = null) {
    const response = await api.post('/api/v1/github/push', {
      path,
      remote_name: remoteName,
      branch,
    })
    return response.data
  },
}


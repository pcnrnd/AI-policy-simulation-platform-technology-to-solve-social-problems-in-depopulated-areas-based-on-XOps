import api from './api'

/**
 * MinIO 관련 API 서비스
 */
export const minioService = {
  /**
   * 버킷 목록 조회
   */
  async listBuckets() {
    const response = await api.get('/api/v1/minio/buckets')
    return response.data
  },

  /**
   * 버킷 생성
   * @param {string} bucketName - 버킷 이름
   */
  async createBucket(bucketName) {
    const response = await api.post('/api/v1/minio/buckets', null, {
      params: { bucket_name: bucketName },
    })
    return response.data
  },

  /**
   * 파일 업로드 (Multipart Form)
   * @param {string} bucketName - 버킷 이름
   * @param {string} objectName - 객체 이름 (파일 경로)
   * @param {File} file - 업로드할 파일
   */
  async uploadFile(bucketName, objectName, file) {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post(
      `/api/v1/minio/upload?bucket_name=${bucketName}&object_name=${objectName}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )
    return response.data
  },

  /**
   * 로컬 파일 경로로부터 업로드
   * @param {string} bucketName - 버킷 이름
   * @param {string} objectName - 객체 이름 (파일 경로)
   * @param {string} filePath - 로컬 파일 경로
   * @param {string} contentType - 콘텐츠 타입 (선택)
   */
  async uploadFileFromPath(bucketName, objectName, filePath, contentType = null) {
    const response = await api.post('/api/v1/minio/upload-path', {
      bucket_name: bucketName,
      object_name: objectName,
      file_path: filePath,
      content_type: contentType,
    })
    return response.data
  },

  /**
   * 버킷 내 객체 목록 조회
   * @param {string} bucketName - 버킷 이름
   * @param {string} prefix - 객체 이름 접두사 (선택)
   */
  async listObjects(bucketName, prefix = null) {
    const response = await api.get('/api/v1/minio/objects', {
      params: {
        bucket_name: bucketName,
        prefix,
      },
    })
    return response.data
  },

  /**
   * 파일 다운로드
   * @param {string} bucketName - 버킷 이름
   * @param {string} objectName - 객체 이름
   */
  async downloadFile(bucketName, objectName) {
    const response = await api.post('/api/v1/minio/download', {
      bucket_name: bucketName,
      object_name: objectName,
    })
    return response.data
  },

  /**
   * 파일 삭제
   * @param {string} bucketName - 버킷 이름
   * @param {string} objectName - 객체 이름
   */
  async deleteFile(bucketName, objectName) {
    const response = await api.delete('/api/v1/minio/delete', {
      data: {
        bucket_name: bucketName,
        object_name: objectName,
      },
    })
    return response.data
  },
}


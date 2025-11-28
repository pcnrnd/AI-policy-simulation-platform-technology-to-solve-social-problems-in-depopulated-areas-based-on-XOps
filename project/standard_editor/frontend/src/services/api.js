import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
})

// 요청 인터셉터
api.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 응답 인터셉터
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    if (error.response) {
      // 서버에서 응답이 왔지만 에러 상태 코드
      console.error('API Error:', error.response.data)
    } else if (error.request) {
      // 요청은 보냈지만 응답이 없음
      console.error('Network Error:', error.request)
    } else {
      // 요청 설정 중 에러
      console.error('Error:', error.message)
    }
    return Promise.reject(error)
  }
)

export default api


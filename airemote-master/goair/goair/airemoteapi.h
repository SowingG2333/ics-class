#ifndef AIREMOTE_API_H
#define AIREMOTE_API_H
 
#ifdef __cplusplus
extern "C" {
#endif

void* create_air();
int use_remote(void *air, const char *remote);
int inference_remote(void *air, void *input, size_t size, void *result, size_t *len);
void destroy_air(void *air);

#ifdef __cplusplus
}
#endif

#endif // AIREMOTE_API_H

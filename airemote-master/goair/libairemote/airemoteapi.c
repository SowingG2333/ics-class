#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "airemoteapi.h"
#include "googlenet-dummy.c"
#include "picasso-dummy.c"

static char g_remote[50];

void *create_air()
{
	printf(">>> AI Remote Created!\n");
}

int use_remote(void *air, const char *remote)
{
	printf(">>> Use remote...\n");
	printf("%s\n", remote);
	memcpy(g_remote, remote, strlen(remote));
	return 0;
}

int inference_remote(void *air, void *input, size_t size, void *result, size_t *len)
{
	printf(">>> Inference remote...\n");
	if (strcmp(g_remote, "tcp://localhost:5530") == 0)
	{
		memcpy(result, googlenet_dummy_bin, googlenet_dummy_bin_len);
		*len = googlenet_dummy_bin_len;
	}
	else if (strcmp(g_remote, "tcp://localhost:5550") == 0)
	{
		memcpy(result, picasso_dummy_bin, picasso_dummy_bin_len);
		*len = picasso_dummy_bin_len;
	}
	return 0;
}

void destroy_air(void *air)
{
	printf(">>> AI Remote Destoried!\n");
}
